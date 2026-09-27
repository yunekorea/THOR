#!/usr/bin/env python3
"""
THOR_NDP_Host_parallel.py -- N concurrent inferences against the worker-pool target.

The host GPU sits at roughly 11% utilization in the single-stream NDP run,
because it spends most of its time blocked on the target. This runs several
inferences at once so that while stream A waits on a bootstrap, stream B uses
the GPU.

What improves is THROUGHPUT, not latency. Any one inference still takes about as
long as it did before; you get more of them per hour. Report it that way.

Keys are shared across streams -- one HE, one set -- because duplicating them
would give back the VRAM the NDP split saves. Safety comes from the engine lock
in thor.ndp_parallel; see HENDPParallel for why that costs nothing here.

Sizing for 48 GB VRAM: shared keys are about 14 GiB (rotation + relin + conj +
fixed rotation keys, no bootstrap key), and each stream adds its own ciphertexts
plus one stage's weights, roughly 3-4 GiB. Four streams lands near 30 GiB.

    poetry run python THOR_NDP_Host_parallel.py --transport tcp --streams 4
    poetry run python THOR_NDP_Host_parallel.py --transport rdma --streams 4
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)

project_root = os.path.abspath(os.path.join(os.getcwd(), '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from thor.forward import forward_layer, load_encrypted_input
from thor.ndp_parallel import HENDPParallel, make_parallel_transport


def parse_args():
    p = argparse.ArgumentParser(
        description="Concurrent THOR inferences with bootstrapping offloaded.")
    p.add_argument("--streams", type=int, default=4,
                   help="Concurrent inferences. Match the target's --workers.")
    p.add_argument("--dataset-type", default="mrpc")
    p.add_argument("--target-idx", type=int, default=0,
                   help="Validation index for stream 0; stream i uses target-idx + i.")
    p.add_argument("--dataset-path", default="",
                   help="save_to_disk snapshot, e.g. ./datasets/mrpc.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--keys-dir", default=None)
    p.add_argument("--output-dir", default="./ndp_parallel_results")
    p.add_argument("--mode", default="gpu", choices=["gpu", "async gpu"])
    p.add_argument("--transport", default="tcp", choices=["tcp", "rdma"])
    p.add_argument("--host-ip", default="192.168.100.2")
    p.add_argument("--target-ip", default="192.168.100.1")
    p.add_argument("--port", type=int, default=9998, help="TCP only.")
    p.add_argument("--nvme-dev", default="nvme1n1", help="RDMA only.")
    p.add_argument("--ib-dev", default="enp216s0np0", help="RDMA only.")
    p.add_argument("--recv-mode", default="pool", choices=["pool", "per-request"],
                   help="RDMA only. 'pool' pre-posts receive buffers and demuxes "
                        "replies (needed for >1 stream). 'per-request' reproduces "
                        "the working single-stream host exactly -- use with "
                        "--streams 1 to tell host bugs from target bugs.")
    p.add_argument("--quiet-bootstrap", dest="verbose_bootstrap",
                   action="store_false", default=True,
                   help="Do not print a line per bootstrap call. With N streams "
                        "that is N x 294 lines.")
    p.add_argument("--load-bootstrap-key", action="store_true",
                   help="Also load the bootstrap key locally (costs ~17 GiB VRAM).")
    return p.parse_args()


def run_stream(stream_id, x, clear_attention_mask, he, results, lock):
    """One full inference: 12 layers, pooler, classifier."""
    t_start = time.perf_counter()
    layer_seconds = []
    try:
        for layer_idx in range(12):
            t0 = time.perf_counter()
            x, variables = forward_layer(x, layer_idx, clear_attention_mask, he)
            layer_seconds.append(round(time.perf_counter() - t0, 3))
            print(f"  [s{stream_id}] layer {layer_idx} done "
                  f"({layer_seconds[-1]:.1f}s)", flush=True)

        t0 = time.perf_counter()
        x = he.stage_17_pooler(x)
        x = he.stage_18_classifier(x)
        head_seconds = time.perf_counter() - t0

        total = time.perf_counter() - t_start
        with lock:
            results[stream_id] = dict(
                ok=True, total_seconds=round(total, 3),
                layer_seconds=layer_seconds, head_seconds=round(head_seconds, 3),
            )
        print(f"[s{stream_id}] DONE in {total:.1f}s", flush=True)
    except Exception as exc:
        with lock:
            results[stream_id] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        print(f"[s{stream_id}] FAILED: {type(exc).__name__}: {exc}", flush=True)


def main():
    args = parse_args()
    print(args)

    key_size = "medium" if args.compact else "large"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys_dir = Path(args.keys_dir) if args.keys_dir else \
        Path("./keys_desilo") / ("compact" if args.compact else "default")
    params = json.loads((keys_dir / "params.json").read_text())
    if bool(params.get("compact")) != args.compact:
        raise SystemExit(f"{keys_dir} built with compact={params.get('compact')}, "
                         f"run uses compact={args.compact}.")

    print(f"Connecting to target via {args.transport}")
    transport = make_parallel_transport(args.transport, args)
    print("  connected")

    setup_start = time.perf_counter()
    print(f"Setting up engine and keys (mode={args.mode!r}, reading {keys_dir})")
    he = HENDPParallel(args.device, args.compact, key_size,
                       keys_dir=keys_dir, mode=args.mode, transport=transport,
                       skip_bootstrap_key=not args.load_bootstrap_key,
                       verbose=args.verbose_bootstrap)
    print(f"  keys ready ({time.perf_counter() - setup_start:.1f}s, shared by all streams)")

    # One encrypted input per stream -- different validation samples, so this is
    # a genuine multi-request workload rather than the same inference N times.
    inputs = []
    for s in range(args.streams):
        idx = args.target_idx + s
        t0 = time.perf_counter()
        _, x, _, _, mask = load_encrypted_input(
            args.dataset_type, idx, he, dataset_path=args.dataset_path)
        inputs.append((x, mask, idx))
        print(f"  stream {s}: sample {idx} encrypted ({time.perf_counter() - t0:.1f}s)")
    setup_seconds = time.perf_counter() - setup_start

    results, lock = {}, threading.Lock()
    print(f"\nStarting {args.streams} concurrent inferences at {datetime.now()}")
    run_start = time.perf_counter()

    # thread name is what HENDPParallel.bootstrap tags its per-call lines with
    threads = [
        threading.Thread(target=run_stream,
                         args=(s, inputs[s][0], inputs[s][1], he, results, lock),
                         name=f"s{s}")
        for s in range(args.streams)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall = time.perf_counter() - run_start
    ok = [r for r in results.values() if r.get("ok")]
    bs = he.report()

    print()
    print(f"Wall time for {args.streams} inferences: {wall:.1f}s")
    if ok:
        lat = [r["total_seconds"] for r in ok]
        print(f"  completed          : {len(ok)}/{args.streams}")
        print(f"  per-inference      : min {min(lat):.1f}s  max {max(lat):.1f}s  "
              f"mean {sum(lat)/len(lat):.1f}s")
        print(f"  THROUGHPUT         : {len(ok) / wall * 3600:.2f} inferences/hour")
        print(f"  (single-stream equivalent would be "
              f"{1 / (sum(lat)/len(lat)) * 3600:.2f}/hour)")
    print(f"Bootstraps offloaded : {bs['bootstrap_calls']}")
    print(f"  total offload time : {bs['bootstrap_seconds']:.1f}s summed over streams "
          f"({bs['bootstrap_mean_seconds']:.2f}s each)")
    print(f"  serialize/deserial : {bs['serialize_seconds']:.1f}s")
    print(f"  transport + remote : {bs['transport_seconds']:.1f}s")
    print(f"  traffic            : {bs['mib_sent']:.0f} MiB out, "
          f"{bs['mib_received']:.0f} MiB back")
    if bs["bootstrap_per_stream"]:
        print("  per stream:")
        for name, v in bs["bootstrap_per_stream"].items():
            print(f"    {name:4s} {v['calls']:4d} calls  {v['seconds']:8.1f}s  "
                  f"({v['mean_seconds']:.2f}s each)")
    # Summed offload time exceeds wall time whenever streams overlap -- that
    # ratio is the concurrency actually achieved.
    if wall:
        print(f"  overlap factor     : {bs['bootstrap_seconds'] / wall:.2f}x "
              f"(1.0 = no concurrency, {args.streams}.0 = perfect)")

    (output_dir / "ndp_parallel_result.json").write_text(json.dumps(dict(
        streams=args.streams, dataset_type=args.dataset_type,
        dataset_path=args.dataset_path or None,
        target_idx=args.target_idx, compact=args.compact, key_size=key_size,
        mode=args.mode, transport=args.transport, keys_dir=str(keys_dir),
        setup_seconds=round(setup_seconds, 3),
        wall_seconds=round(wall, 3),
        completed=len(ok),
        throughput_per_hour=round(len(ok) / wall * 3600, 3) if wall else 0,
        per_stream=results,
        **bs,
    ), indent=2) + "\n")
    print(f"Wrote {output_dir / 'ndp_parallel_result.json'}")
    transport.close()


if __name__ == "__main__":
    main()