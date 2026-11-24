#!/usr/bin/env python3
"""
THOR_baseline.py -- forward.ipynb, ported to the DESILO-FHE THOR.

Mirrors the notebook section by section:

    1.   Experiment setup (engine + keys)
    1-2. Load and encrypt data
    1-3. Load and run plain model .............. SKIPPED
    1-4. Load HE model ......................... now inside HE, nothing to do
    2.   Forward 12 attention layers ........... no plotting
    3.   Run pooler and classification
    4.   Comparison with the actual label ...... SKIPPED

Sections 1-3 and 4 are deliberately absent, so this script never imports
transformers-for-plain-inference, matplotlib, or `datasets`.

The notebook's per-layer boilerplate

    layer_idx = 0
    thor_attention = thor_bert.attentions[layer_idx]
    thor_ff = thor_bert.ffs[layer_idx]
    x1, variables = forward_layer(x)

is gone: `forward_layer` now takes `layer_idx` and pulls its own weights, and
each layer feeds the previous one's output back into the same `x`, so no chain
of x1..x12 is kept alive.

Usage:

    poetry run python generate_keys.py          # once
    poetry run python THOR_baseline.py
    poetry run python THOR_baseline.py --compact --target-idx 3
    poetry run python THOR_baseline.py --inferences 4    # serial GPU-only reference

--inferences N runs N inferences one after another in this one process, on
samples target-idx .. target-idx+N-1, sharing the engine and keys -- the
GPU-only serial counterpart of THOR_NDP_Host_parallel.py / THOR_Local_Main.py
with --streams N (same samples, all inputs encrypted during setup, wall time
measured from the end of setup). With the default N=1 the run is unchanged.
"""

import os, sys
project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
project_root = os.path.abspath(os.path.join(os.getcwd(), '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import gc
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from thor.forward import forward_layer, load_encrypted_input
from thor.he import HE
from thor.timer import Timer


class BootstrapTimer:
    """Times every HE.bootstrap() call, for comparison against the NDP host.

    Patched onto the INSTANCE, not the class: he.py calls `self.bootstrap(...)`
    internally from he_inv, he_invsqrt, stage_07_softmax and stage_13, and Python
    resolves that through the instance dict first -- so this catches the internal
    calls too, not just any visible in forward_layer. HENDP.bootstrap overrides
    the same method to ship the ciphertext to the target, so the `bootstrap_calls`
    and `bootstrap_seconds` reported here line up with its report() fields.
    """

    def __init__(self, he, verbose=True):
        self.calls = 0
        self.seconds = 0.0
        self.per_layer = {}
        self._layer = None
        self.verbose = verbose
        self._orig = he.bootstrap
        he.bootstrap = self._call

    def set_layer(self, layer):
        self._layer = layer

    def _call(self, ciphertext):
        self.calls += 1
        n = self.calls
        level_in = getattr(ciphertext, "level", None)
        t0 = time.perf_counter()
        out = self._orig(ciphertext)
        dt = time.perf_counter() - t0
        self.seconds += dt
        if self.verbose:
            print(f"  [bs #{n:04d}] local bootstrap in {dt:.2f}s "
                  f"(level {level_in} -> {getattr(out, 'level', None)})")
        if self._layer is not None:
            rec = self.per_layer.setdefault(self._layer, {"calls": 0, "seconds": 0.0})
            rec["calls"] += 1
            rec["seconds"] += dt
        return out

    def report(self):
        return dict(
            bootstrap_calls=self.calls,
            bootstrap_seconds=round(self.seconds, 3),
            bootstrap_mean_seconds=round(self.seconds / self.calls, 3) if self.calls else 0.0,
            # keys are the 12 int layer indices plus the string "head"; a bare
            # sorted() raises TypeError on the mixed types.
            bootstrap_per_layer={
                str(k): {"calls": v["calls"], "seconds": round(v["seconds"], 3)}
                for k, v in sorted(self.per_layer.items(),
                                   key=lambda kv: (isinstance(kv[0], str), kv[0]))
            },
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the THOR encrypted forward pass (no plain reference, no plots)."
    )
    parser.add_argument("--dataset-type", default="mrpc")
    parser.add_argument("--target-idx", type=int, default=0,
                        help="Validation sample index to run.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--compact", action="store_true",
                        help="Memory-optimized mode (medium bootstrap key).")
    parser.add_argument("--keys-dir", default=None,
                        help="Key set from generate_keys.py. Default "
                             "./keys_desilo/default, or ./keys_desilo/compact.")
    parser.add_argument("--no-load-keys", dest="load_keys", action="store_false",
                        default=True,
                        help="Generate keys in-process instead of reading them.")
    parser.add_argument("--dataset-path", default="",
                        help="Load the dataset from a save_to_disk snapshot instead of the HuggingFace Hub, e.g. ./datasets/mrpc. Use this to run against the same samples as the Liberate results.")
    parser.add_argument("--mode", default="gpu", choices=["gpu", "async gpu"],
                        help="Engine mode. Defaults to 'gpu' to match "
                             "THOR_NDP_Host.py -- HE's own default is 'async gpu', "
                             "which is faster, so leaving it unset would make the "
                             "baseline win on engine mode rather than on where "
                             "bootstrapping runs.")
    parser.add_argument("--no-rotation-key", dest="use_rotation_key",
                        action="store_false", default=True,
                        help="Do not load the general rotation key. The baseline "
                             "loads it by default so its key I/O and rotation path "
                             "match the NDP host; rotations then come from the "
                             "bootstrap key, as stock THOR does.")
    parser.add_argument("--quiet-bootstrap", dest="verbose_bootstrap",
                        action="store_false", default=True,
                        help="Do not print a line per bootstrap call.")
    parser.add_argument("--output-dir", default="./baseline_results")
    parser.add_argument("--print-rotate-levels", action="store_true")
    parser.add_argument("--inferences", type=int, default=1, metavar="N",
                        help="Run N inferences one after another (samples "
                             "target-idx .. target-idx+N-1) with one engine and one "
                             "key set: the serial GPU-only reference for the "
                             "N-stream NDP / local runs. Default 1 = unchanged.")
    return parser.parse_args()


def gpu_used_mib(device):
    """Device-wide used VRAM in MiB via nvidia-smi, or None.

    nvidia-smi rather than torch.cuda.mem_get_info: torch is only used on the
    CPU here, and asking it would create a CUDA context (hundreds of MiB) on a
    GPU this run already fills with keys.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", str(device)],
            capture_output=True, text=True, timeout=10).stdout.strip()
        used, total = (int(v) for v in out.split(","))
        return used, total
    except Exception:
        return None


def run_inference(item, he, timer, bootstrap_timer):
    """One full inference: 12 layers, pooler, classifier (the original main body).

    The input ciphertext is popped out of `item`, so once layer 0 has replaced
    `x` nothing references the input any more -- it is freed at the same point
    as in the original single-run code, not held for the whole inference.
    """
    x = item.pop("x")
    clear_attention_mask = item["mask"]
    layer_seconds = []
    for layer_idx in range(12):
        print(f"Forwarding layer #{layer_idx}")
        bootstrap_timer.set_layer(layer_idx)
        t0 = time.perf_counter()
        with timer.layer(layer_idx):
            x, variables = forward_layer(x, layer_idx, clear_attention_mask, he)
        elapsed = time.perf_counter() - t0
        layer_seconds.append(round(elapsed, 3))
        print(f"  layer #{layer_idx} done ({elapsed:.1f}s)")
    del variables

    print("all layers done")
    print("now:", datetime.now())
    timer.print_legend()

    # ---- 3. Run pooler and classification -----------------------------------
    bootstrap_timer.set_layer("head")
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
    del x
    return layer_seconds, pooler_seconds, classifier_seconds


def main():
    args = parse_args()
    print(args)
    if args.inferences < 1:
        raise SystemExit("--inferences must be at least 1.")

    key_size = "medium" if args.compact else "large"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys_dir = None
    if args.load_keys:
        keys_dir = Path(args.keys_dir) if args.keys_dir else \
            Path("./keys_desilo") / ("compact" if args.compact else "default")
        # generate_keys.py records what it built these for. Worth one check:
        # desilofhe's read_*_key does not verify a key against the engine
        # loading it, so a mismatched key set gives wrong plaintexts, not an error.
        params = json.loads((keys_dir / "params.json").read_text())
        if bool(params.get("compact")) != args.compact:
            raise SystemExit(
                f"{keys_dir} was built with compact={params.get('compact')}, "
                f"but this run uses compact={args.compact}. Regenerate with "
                f"generate_keys.py using the same flag."
            )

    timer = Timer()
    started = time.perf_counter()

    # ---- 1. Experiment setup, 1-2. Load and encrypt data --------------------
    with timer.setup():
        print(f"Setting up engine and keys "
              f"({'reading ' + str(keys_dir) if keys_dir else 'generating in-process'})")
        t0 = time.perf_counter()
        # mode: see --mode help. use_rotation_key: load the rotation key as well
        # as the bootstrap key, so the baseline reads the same key bytes the NDP
        # host does plus the bootstrap key, and serves the bootstrap-delta
        # rotations from the same key the NDP host uses. The only remaining
        # difference between the two runs is where bootstrapping executes.
        he = HE(args.device, args.compact, key_size, timer, keys_dir=keys_dir,
                mode=args.mode, use_rotation_key=args.use_rotation_key)
        rot = "rotation_key" if args.use_rotation_key else "bootstrap_key"
        print(f"  keys ready ({time.perf_counter() - t0:.1f}s, mode={args.mode!r}, "
              f"bootstrap key loaded, rotations served by {rot})")

        bootstrap_timer = BootstrapTimer(he, verbose=args.verbose_bootstrap)

        # All inputs are encrypted here, inside setup, exactly as the N-stream
        # NDP / local hosts do before their wall clock starts.
        inputs = []
        for i in range(args.inferences):
            idx = args.target_idx + i
            print("Encrypting input" + (f" {i} (sample {idx})" if args.inferences > 1 else ""))
            t0 = time.perf_counter()
            _, x, _, _, clear_attention_mask = load_encrypted_input(
                args.dataset_type, idx, he, dataset_path=args.dataset_path
            )
            inputs.append(dict(sample=idx, x=x, mask=clear_attention_mask))
            print(f"  input encrypted ({time.perf_counter() - t0:.1f}s)")
        del x, clear_attention_mask

    setup_seconds = time.perf_counter() - started
    vram = gpu_used_mib(args.device)
    if vram:
        print(f"  VRAM after setup: {vram[0] / 1024:.1f} of {vram[1] / 1024:.1f} GiB used")

    # ---- 2. Forward 12 attention layers, 3. pooler + classifier ------------
    run_start = time.perf_counter()
    per_inference = []
    for i in range(args.inferences):
        item, inputs[i] = inputs[i], None      # run_inference pops x out of item
        idx = item["sample"]
        if args.inferences > 1:
            print(f"\n=== Inference {i + 1}/{args.inferences} (sample {idx}) "
                  f"starting at +{time.perf_counter() - run_start:.1f}s ===")
        calls0, secs0 = bootstrap_timer.calls, bootstrap_timer.seconds
        t_begin = time.perf_counter()
        layer_s, pooler_s, classifier_s = run_inference(item, he, timer, bootstrap_timer)
        t_end = time.perf_counter()
        del item
        gc.collect()
        vram = gpu_used_mib(args.device)
        rec = dict(
            sample=idx,
            start_offset=round(t_begin - run_start, 3),
            seconds=round(t_end - t_begin, 3),
            layer_seconds=layer_s,
            pooler_seconds=round(pooler_s, 3),
            classifier_seconds=round(classifier_s, 3),
            bootstrap_calls=bootstrap_timer.calls - calls0,
            bootstrap_seconds=round(bootstrap_timer.seconds - secs0, 3),
            vram_used_mib_after=vram[0] if vram else None,
        )
        per_inference.append(rec)
        if args.inferences > 1:
            print(f"=== Inference {i + 1}/{args.inferences} done in {rec['seconds']:.1f}s "
                  f"({rec['bootstrap_calls']} bootstraps, {rec['bootstrap_seconds']:.1f}s"
                  + (f"; VRAM {vram[0] / 1024:.1f} GiB used" if vram else "") + ") ===")
    run_seconds = time.perf_counter() - run_start

    layer_seconds = [s for r in per_inference for s in r["layer_seconds"]]
    pooler_seconds = sum(r["pooler_seconds"] for r in per_inference)
    classifier_seconds = sum(r["classifier_seconds"] for r in per_inference)

    total = time.perf_counter() - started
    print("now:", datetime.now())
    timer.print_legend()

    if args.print_rotate_levels:
        for delta, level in he.rotate_levels.items():
            print(f"Rotate delta {delta} max level {level}")

    bs = bootstrap_timer.report()
    print()
    print(f"Total wall time: {total:.1f}s "
          f"(12 layers: {sum(layer_seconds):.1f}s, "
          f"pooler+classifier: {pooler_seconds + classifier_seconds:.1f}s)")
    print(f"Bootstraps (local)   : {bs['bootstrap_calls']}")
    print(f"  total time         : {bs['bootstrap_seconds']:.1f}s "
          f"({bs['bootstrap_mean_seconds']:.2f}s each)")
    if total:
        print(f"  share of wall time : {100 * bs['bootstrap_seconds'] / total:.1f}%")

    # Comparable to the "Wall time for N inferences" of the N-stream hosts:
    # measured from the end of setup, so key loading is not in it.
    n = args.inferences
    print(f"Wall time for {n} inference(s), after setup: {run_seconds:.1f}s "
          f"(setup {setup_seconds:.1f}s)")
    if n > 1:
        lat = [r["seconds"] for r in per_inference]
        print(f"  per-inference      : min {min(lat):.1f}s  max {max(lat):.1f}s  "
              f"mean {sum(lat) / n:.1f}s")
        for k, r in enumerate(per_inference):
            print(f"    #{k} sample {r['sample']:3d}  {r['seconds']:8.1f}s  "
                  f"bootstraps {r['bootstrap_seconds']:7.1f}s  "
                  f"other {r['seconds'] - r['bootstrap_seconds']:7.1f}s"
                  + (f"  VRAM after {r['vram_used_mib_after'] / 1024:.1f} GiB"
                     if r['vram_used_mib_after'] is not None else ""))
    if run_seconds:
        print(f"  THROUGHPUT         : {n / run_seconds * 3600:.2f} inferences/hour")

    (output_dir / "baseline_result.json").write_text(json.dumps(dict(
        dataset_type=args.dataset_type,
        dataset_path=args.dataset_path or None,
        target_idx=args.target_idx,
        device=args.device,
        compact=args.compact,
        key_size=key_size,
        mode=args.mode,
        use_rotation_key=args.use_rotation_key,
        keys_dir=str(keys_dir) if keys_dir else None,
        total_seconds=round(total, 3),
        # With --inferences N > 1: layer_seconds is all N x 12 layers in run
        # order, pooler/classifier are summed; per_inference has the split.
        layer_seconds=layer_seconds,
        pooler_seconds=round(pooler_seconds, 3),
        classifier_seconds=round(classifier_seconds, 3),
        inferences=n,
        setup_seconds=round(setup_seconds, 3),
        wall_seconds=round(run_seconds, 3),
        throughput_per_hour=round(n / run_seconds * 3600, 3) if run_seconds else 0,
        per_inference=per_inference,
        **bs,
    ), indent=2) + "\n")
    print(f"Wrote {output_dir / 'baseline_result.json'}")


if __name__ == "__main__":
    main()