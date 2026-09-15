#!/usr/bin/env python3
"""
THOR_NDP_host.py -- THOR_baseline.py with bootstrapping offloaded to the target.

Identical to THOR_baseline.py except that `HE` is replaced by `HENDP`, whose
bootstrap() serializes the ciphertext and ships it to THOR_NDP_target.py. The
12-layer loop, the pooler and the classifier are unchanged, and no bootstrap
key is loaded on this side.

Start the target first, then:

    poetry run python THOR_NDP_host.py --transport tcp   --target-ip 192.168.100.1
    poetry run python THOR_NDP_host.py --transport rdma
"""

import os, sys
project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
project_root = os.path.abspath(os.path.join(os.getcwd(), '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from thor.forward import forward_layer, load_encrypted_input
from thor.he_ndp import HENDP, make_transport
from thor.timer import Timer


def parse_args():
    p = argparse.ArgumentParser(description="THOR forward pass with offloaded bootstrapping.")
    p.add_argument("--dataset-type", default="mrpc")
    p.add_argument("--target-idx", type=int, default=0)
    p.add_argument("--device", type=int, default=0, help="CUDA device index.")
    p.add_argument("--compact", action="store_true")
    p.add_argument("--keys-dir", default=None)
    p.add_argument("--output-dir", default="./ndp_host_results")
    p.add_argument("--mode", default="gpu", choices=["gpu", "async gpu"],
                   help="Host engine mode. Defaults to 'gpu', not 'async gpu': "
                        "Desilo documents async gpu as experimental and advises "
                        "against sharing its data structures with another engine.")
    p.add_argument("--transport", default="tcp", choices=["tcp", "rdma"])
    p.add_argument("--host-ip", default="192.168.100.2")
    p.add_argument("--target-ip", default="192.168.100.1")
    p.add_argument("--port", type=int, default=9998, help="TCP transport only.")
    p.add_argument("--nvme-dev", default="nvme1n1", help="RDMA transport only.")
    p.add_argument("--ib-dev", default="enp216s0np0", help="RDMA transport only.")
    p.add_argument("--load-bootstrap-key", action="store_true",
                   help="Also load the bootstrap key locally. Only useful to A/B "
                        "local vs offloaded bootstrapping; wastes VRAM otherwise.")
    return p.parse_args()


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
    transport = make_transport(args.transport, args)
    print("  connected")

    timer = Timer()
    started = time.perf_counter()

    with timer.setup():
        print(f"Setting up engine and keys (mode={args.mode!r}, reading {keys_dir})")
        t0 = time.perf_counter()
        he = HENDP(args.device, args.compact, key_size, timer,
                   keys_dir=keys_dir, mode=args.mode, transport=transport,
                   skip_bootstrap_key=not args.load_bootstrap_key)
        print(f"  keys ready ({time.perf_counter() - t0:.1f}s"
              f"{', bootstrap key NOT loaded' if not args.load_bootstrap_key else ''})")

        print("Encrypting input")
        t0 = time.perf_counter()
        _, x, _, _, clear_attention_mask = load_encrypted_input(
            args.dataset_type, args.target_idx, he
        )
        print(f"  input encrypted ({time.perf_counter() - t0:.1f}s)")

    layer_seconds = []
    for layer_idx in range(12):
        print(f"Forwarding layer #{layer_idx}")
        t0 = time.perf_counter()
        with timer.layer(layer_idx):
            x, variables = forward_layer(x, layer_idx, clear_attention_mask, he)
        elapsed = time.perf_counter() - t0
        layer_seconds.append(round(elapsed, 3))
        print(f"  layer #{layer_idx} done ({elapsed:.1f}s)")

    print("all layers done")
    print("now:", datetime.now())
    timer.print_legend()

    print("Running pooler")
    t0 = time.perf_counter()
    with timer.stage(17, "pooler"):
        x = he.stage_17_pooler(x)
    pooler_seconds = time.perf_counter() - t0
    print(f"  pooler done ({pooler_seconds:.1f}s)")

    print("Running classifier")
    t0 = time.perf_counter()
    with timer.stage(18, "classifier"):
        x = he.stage_18_classifier(x)
    classifier_seconds = time.perf_counter() - t0
    print(f"  classifier done ({classifier_seconds:.1f}s)")

    total = time.perf_counter() - started
    timer.print_legend()

    bs = he.report()
    print()
    print(f"Total wall time: {total:.1f}s (12 layers: {sum(layer_seconds):.1f}s)")
    print(f"Bootstraps offloaded : {bs['bootstrap_calls']}")
    print(f"  total offload time : {bs['bootstrap_seconds']:.1f}s "
          f"({bs['bootstrap_seconds']/max(bs['bootstrap_calls'],1):.2f}s each)")
    print(f"  serialize/deserial : {bs['serialize_seconds']:.1f}s")
    print(f"  transport + remote : {bs['transport_seconds']:.1f}s")
    print(f"  traffic            : {bs['mib_sent']:.0f} MiB out, "
          f"{bs['mib_received']:.0f} MiB back")

    (output_dir / "ndp_host_result.json").write_text(json.dumps(dict(
        dataset_type=args.dataset_type, target_idx=args.target_idx,
        compact=args.compact, key_size=key_size, mode=args.mode,
        transport=args.transport, keys_dir=str(keys_dir),
        total_seconds=round(total, 3), layer_seconds=layer_seconds,
        pooler_seconds=round(pooler_seconds, 3),
        classifier_seconds=round(classifier_seconds, 3),
        **bs,
    ), indent=2) + "\n")
    print(f"Wrote {output_dir / 'ndp_host_result.json'}")
    transport.close()


if __name__ == "__main__":
    main()