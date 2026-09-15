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

    def __init__(self, he):
        self.calls = 0
        self.seconds = 0.0
        self.per_layer = {}
        self._layer = None
        self._orig = he.bootstrap
        he.bootstrap = self._call

    def set_layer(self, layer):
        self._layer = layer

    def _call(self, ciphertext):
        self.calls += 1
        t0 = time.perf_counter()
        out = self._orig(ciphertext)
        dt = time.perf_counter() - t0
        self.seconds += dt
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
    parser.add_argument("--output-dir", default="./baseline_results")
    parser.add_argument("--print-rotate-levels", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

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
        he = HE(args.device, args.compact, key_size, timer, keys_dir=keys_dir)
        print(f"  keys ready ({time.perf_counter() - t0:.1f}s)")

        bootstrap_timer = BootstrapTimer(he)

        print("Encrypting input")
        t0 = time.perf_counter()
        _, x, _, _, clear_attention_mask = load_encrypted_input(
            args.dataset_type, args.target_idx, he, dataset_path=args.dataset_path
        )
        print(f"  input encrypted ({time.perf_counter() - t0:.1f}s)")

    # ---- 2. Forward 12 attention layers -------------------------------------
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

    (output_dir / "baseline_result.json").write_text(json.dumps(dict(
        dataset_type=args.dataset_type,
        dataset_path=args.dataset_path or None,
        target_idx=args.target_idx,
        device=args.device,
        compact=args.compact,
        key_size=key_size,
        keys_dir=str(keys_dir) if keys_dir else None,
        total_seconds=round(total, 3),
        layer_seconds=layer_seconds,
        pooler_seconds=round(pooler_seconds, 3),
        classifier_seconds=round(classifier_seconds, 3),
        **bs,
    ), indent=2) + "\n")
    print(f"Wrote {output_dir / 'baseline_result.json'}")


if __name__ == "__main__":
    main()