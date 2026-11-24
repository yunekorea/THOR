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

Two locking modes (see thor.ndp_parallel):
    default      the engine lock is taken per engine call -- streams interleave
                 call by call
    --gpu-token  a stream keeps the lock for its whole GPU phase and hands it
                 over only while a bootstrap is in flight or its next weights
                 are being read from disk

Every stage of every layer is timed per stream (StageTimer): one line per stage
as it finishes, tables at the end, and stage_times.csv in --output-dir.

Sizing for 48 GB VRAM: shared keys are about 14 GiB (rotation + relin + conj +
fixed rotation keys, no bootstrap key), and each stream adds its own ciphertexts
plus one stage's weights, roughly 3-4 GiB. Four streams lands near 30 GiB.

    poetry run python THOR_NDP_Host_parallel.py --transport rdma --streams 4
    poetry run python THOR_NDP_Host_parallel.py --transport rdma --streams 2 --gpu-token
    poetry run python THOR_NDP_Host_parallel.py --transport rdma --streams 4 --gpu-token --stagger 3

--stagger S starts stream i at +i*S seconds instead of all at once, so the
streams' bootstraps do not line up and leave the GPU with nothing to do.

GPU idle time is reported for every run, two ways:
  token idle   (--gpu-token only) exact: time during which no stream holds the
               GPU token, i.e. every stream that has started is waiting on a
               bootstrap or reading weights. Split into ramp (streams still
               starting / already finishing) and steady state (all running).
  nvidia-smi   hardware view, sampled every --gpu-sample-ms: mean utilization,
               idle estimated as sum(1 - util) * interval, and peak VRAM.
Both timelines are written to --output-dir for plotting.
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
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from thor.forward import forward_layer, load_encrypted_input
from thor.ndp_parallel import HENDPParallel, make_parallel_transport
from thor.stage_timer import StageTimer


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
    p.add_argument("--transport", default="rdma", choices=["rdma"],
                   help="RDMA is the only transport (the TCP fallback was removed); "
                        "the flag is kept so scripts passing --transport rdma still work.")
    p.add_argument("--host-ip", default="192.168.100.2")
    p.add_argument("--target-ip", default="192.168.100.1")
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
    p.add_argument("--gpu-token", action="store_true",
                   help="Hold the engine lock for a whole GPU phase instead of per "
                        "engine call. A stream hands the GPU to another stream only "
                        "(1) while its offloaded bootstrap is in flight and (2) while "
                        "its next weights are being read from disk. Default: the "
                        "original per-call locking, so the two can be A/B compared.")
    p.add_argument("--prefetch-threads", type=int, default=1,
                   help="With --gpu-token: threads used to read a stage's weight "
                        "files into the page cache during the handoff (default 1, "
                        "the same sequential pattern as the engine's own reads).")
    p.add_argument("--quiet-stages", action="store_true",
                   help="Do not print a line per stage as it finishes (the tables "
                        "at the end and stage_times.csv are still produced).")
    p.add_argument("--stagger", type=float, default=0.0, metavar="SECONDS",
                   help="Start stream i at +i*SECONDS instead of all at once "
                        "(e.g. 3 -> streams start at +0, +3, +6, +9 s). The wall "
                        "time still runs from the first stream's start, so the "
                        "delay is counted against throughput. Default 0.")
    p.add_argument("--gpu-sample-ms", type=int, default=1000, metavar="MS",
                   help="nvidia-smi sampling interval for GPU utilization / VRAM "
                        "during the run (0 disables it). Default 1000.")
    return p.parse_args()


# ======================================================================
# GPU idle measurement
# ======================================================================
class TokenIdleTracker:
    """Exact busy/idle timeline of the --gpu-token GpuLock.

    The GPU has work exactly while some stream owns the token: a stream gives
    it up only while its bootstrap is in flight or its weights are being read,
    and takes it back before touching the engine again. So "no owner" is the
    application-level definition of an idle GPU.

    Every write to the lock's _owner (in _take, acquire, release, yielded) is
    caught by swapping the lock's class for a subclass where _owner is a
    property -- thor.ndp_parallel itself is not modified.
    """

    def __init__(self, lock):
        self.events = []                     # (perf_counter, busy)
        tracker = self
        base = type(lock)

        class TrackedGpuLock(base):
            @property
            def _owner(self):
                return self.__dict__.get("_tracked_owner")

            @_owner.setter
            def _owner(self, value):
                old = self.__dict__.get("_tracked_owner")
                self.__dict__["_tracked_owner"] = value
                if (old is None) != (value is None):
                    tracker.events.append((time.perf_counter(), value is not None))

        current = lock.__dict__.pop("_owner", None)
        lock.__class__ = TrackedGpuLock
        lock.__dict__["_tracked_owner"] = current

    def idle(self, t0, t1, min_gap=0.01):
        """Idle seconds in [t0, t1], number of gaps >= min_gap, longest gap."""
        if t1 <= t0:
            return dict(seconds=0.0, window=0.0, fraction=0.0, gaps=0, longest=0.0)
        busy = False
        for t, b in self.events:              # state at t0
            if t > t0:
                break
            busy = b
        idle_total, gaps, longest = 0.0, 0, 0.0
        cur = t0
        for t, b in self.events:
            if t <= t0:
                continue
            if t >= t1:
                break
            if not busy and b:                # an idle stretch ends at t
                gap = t - cur
                idle_total += gap
                if gap >= min_gap:
                    gaps += 1
                longest = max(longest, gap)
            if b != busy:
                busy, cur = b, t
        if not busy:
            gap = t1 - cur
            idle_total += gap
            if gap >= min_gap:
                gaps += 1
            longest = max(longest, gap)
        w = t1 - t0
        return dict(seconds=round(idle_total, 3), window=round(w, 3),
                    fraction=round(idle_total / w, 4), gaps=gaps,
                    longest=round(longest, 3))

    def write_csv(self, path, t_ref):
        with open(path, "w") as f:
            f.write("t_seconds,busy\n")
            for t, b in self.events:
                if t >= t_ref:
                    f.write(f"{t - t_ref:.4f},{int(b)}\n")
        return path


class SmiSampler:
    """nvidia-smi utilization / VRAM samples, timestamped on this clock."""

    def __init__(self, device, interval_ms):
        self.interval = interval_ms / 1000.0
        self.samples = []                     # (perf_counter, util %, MiB used)
        self.proc = None
        if interval_ms <= 0:
            return
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", "-i", str(device),
                 "-lms", str(interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except (FileNotFoundError, OSError) as exc:
            print(f"  (nvidia-smi sampling disabled: {exc})")
            return
        threading.Thread(target=self._read, daemon=True, name="smi").start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                util, mem = (int(v) for v in line.strip().split(","))
            except ValueError:
                continue
            self.samples.append((time.perf_counter(), util, mem))

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def summary(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return None
        utils = [u for _, u, _ in s]
        return dict(samples=len(s),
                    mean_util_percent=round(sum(utils) / len(utils), 1),
                    idle_seconds_est=round(sum(100 - u for u in utils) / 100 * self.interval, 1),
                    idle_fraction_est=round(sum(100 - u for u in utils) / 100 / len(utils), 4),
                    zero_util_samples=sum(1 for u in utils if u == 0),
                    peak_vram_mib=max(m for _, _, m in s),
                    mean_vram_mib=round(sum(m for _, _, m in s) / len(s)))

    def write_csv(self, path, t_ref):
        with open(path, "w") as f:
            f.write("t_seconds,util_percent,vram_mib\n")
            for t, u, m in self.samples:
                f.write(f"{t - t_ref:.3f},{u},{m}\n")
        return path


def run_stream(stream_id, x, clear_attention_mask, he, results, lock, run_start):
    """One full inference: 12 layers, pooler, classifier.

    he.stream_turn(): with --gpu-token the stream takes the GPU token here and
    holds it for the whole inference, giving it up only inside bootstraps and
    weight prefetches (and for good if it fails). Without --gpu-token it does
    nothing and locking stays per engine call.
    """
    timer = he.timer
    t_start = time.perf_counter()
    layer_seconds = []
    with he.stream_turn():
        try:
            for layer_idx in range(12):
                t0 = time.perf_counter()
                with timer.layer(layer_idx):   # prints the layer's breakdown line
                    x, variables = forward_layer(x, layer_idx, clear_attention_mask, he)
                layer_seconds.append(round(time.perf_counter() - t0, 3))

            t0 = time.perf_counter()
            with timer.stage(17, "pooler"):
                x = he.stage_17_pooler(x)
            with timer.stage(18, "classifier"):
                x = he.stage_18_classifier(x)
            head_seconds = time.perf_counter() - t0

            t_end = time.perf_counter()
            total = t_end - t_start
            with lock:
                results[stream_id] = dict(
                    ok=True, total_seconds=round(total, 3),
                    start_offset=round(t_start - run_start, 3),
                    end_offset=round(t_end - run_start, 3),
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
    timer = StageTimer(live=not args.quiet_stages)
    he = HENDPParallel(args.device, args.compact, key_size,
                       keys_dir=keys_dir, mode=args.mode, transport=transport,
                       skip_bootstrap_key=not args.load_bootstrap_key,
                       verbose=args.verbose_bootstrap, timer=timer,
                       gpu_token=args.gpu_token,
                       prefetch_threads=args.prefetch_threads)
    lock_desc = ("GPU token (handoff only at bootstraps and weight loading)"
                 if args.gpu_token else "per engine call")
    print(f"  engine lock: {lock_desc}")
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

    token_tracker = TokenIdleTracker(he.lock) if args.gpu_token else None
    smi = SmiSampler(args.device, args.gpu_sample_ms)

    results, lock = {}, threading.Lock()
    stagger_desc = f", staggered by {args.stagger:g}s" if args.stagger > 0 else ""
    print(f"\nStarting {args.streams} concurrent inferences{stagger_desc} at {datetime.now()}")
    run_start = time.perf_counter()

    # thread name is what HENDPParallel.bootstrap tags its per-call lines with
    threads = [
        threading.Thread(target=run_stream,
                         args=(s, inputs[s][0], inputs[s][1], he, results, lock,
                               run_start),
                         name=f"s{s}")
        for s in range(args.streams)
    ]
    for s, t in enumerate(threads):
        if s and args.stagger > 0:
            delay = run_start + s * args.stagger - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        if args.stagger > 0:
            print(f"[s{s}] starting at +{time.perf_counter() - run_start:.1f}s", flush=True)
        t.start()
    for t in threads:
        t.join()

    run_end = time.perf_counter()
    wall = run_end - run_start
    smi.stop()
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

    # How much each stream lost waiting for the others -- the number that tells
    # the two lock modes apart.
    print(f"Waiting for other streams ({bs['lock_mode']} locking):")
    for name, v in bs["waiting_per_stream"].items():
        if args.gpu_token:
            print(f"    {name:4s} {v['handoffs']:4d} handoffs  "
                  f"token wait after bootstrap {v['token_wait_bootstrap']:8.1f}s  "
                  f"after weight prefetch {v['token_wait_weights']:7.1f}s  "
                  f"(longest {v['token_wait_max']:.1f}s)  "
                  f"prefetched {v['prefetch_gib']:.1f} GiB in "
                  f"{v['prefetch_seconds']:.1f}s")
        else:
            print(f"    {name:4s} engine-lock waits {v['engine_lock_waits']:7d}  "
                  f"total {v['engine_lock_wait']:8.1f}s")

    # ---- GPU idle --------------------------------------------------------
    gpu_idle = dict(stagger_seconds=args.stagger)
    starts = [r["start_offset"] for r in ok]
    ends = [r["end_offset"] for r in ok]
    print(f"GPU idle (stagger {args.stagger:g}s):")
    if token_tracker is not None:
        full = token_tracker.idle(run_start, run_end)
        gpu_idle["token_full"] = full
        print(f"  token idle (no stream holds the GPU): {full['seconds']:.1f}s of "
              f"{full['window']:.1f}s wall ({100 * full['fraction']:.1f}%), "
              f"{full['gaps']} gaps >= 10 ms, longest {full['longest']:.1f}s")
        if len(ok) == args.streams and args.streams > 1:
            s0, s1 = run_start + max(starts), run_start + min(ends)
            steady = token_tracker.idle(s0, s1)
            gpu_idle["token_steady"] = dict(steady, start_offset=round(max(starts), 3),
                                            end_offset=round(min(ends), 3))
            ramp = round(full["seconds"] - steady["seconds"], 3)
            ramp_w = round(full["window"] - steady["window"], 3)
            gpu_idle["token_ramp"] = dict(seconds=ramp, window=ramp_w)
            print(f"    steady state (all {args.streams} running, "
                  f"+{max(starts):.0f}s .. +{min(ends):.0f}s): {steady['seconds']:.1f}s of "
                  f"{steady['window']:.1f}s ({100 * steady['fraction']:.1f}%), "
                  f"longest gap {steady['longest']:.1f}s")
            print(f"    ramp-up + ramp-down             : {ramp:.1f}s of {ramp_w:.1f}s")
        tl = token_tracker.write_csv(output_dir / "gpu_token_timeline.csv", run_start)
        print(f"  token timeline: {tl}")
    else:
        print("  token idle: needs --gpu-token (without it the lock is taken per "
              "engine call and does not mark GPU phases)")
    sm = smi.summary(run_start, run_end)
    if sm:
        gpu_idle["nvidia_smi"] = sm
        print(f"  nvidia-smi: mean util {sm['mean_util_percent']:.1f}%, est. idle "
              f"{sm['idle_seconds_est']:.0f}s ({100 * sm['idle_fraction_est']:.1f}%), "
              f"0%-util samples {sm['zero_util_samples']}/{sm['samples']}, "
              f"peak VRAM {sm['peak_vram_mib'] / 1024:.1f} GiB "
              f"(mean {sm['mean_vram_mib'] / 1024:.1f} GiB)")
        sp = smi.write_csv(output_dir / "gpu_smi_samples.csv", run_start)
        print(f"  nvidia-smi samples: {sp}")
    elif args.gpu_sample_ms > 0:
        print("  nvidia-smi: no samples")

    timer.report()
    csv_path = timer.write_csv(output_dir / "stage_times.csv")
    print(f"Wrote {csv_path}")

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
        gpu_token=args.gpu_token, prefetch_threads=args.prefetch_threads,
        stagger_seconds=args.stagger,
        gpu_idle=gpu_idle,
        stage_timing=timer.as_dict(),
        **bs,
    ), indent=2) + "\n")
    print(f"Wrote {output_dir / 'ndp_parallel_result.json'}")
    transport.close()


if __name__ == "__main__":
    main()