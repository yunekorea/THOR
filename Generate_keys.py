#!/usr/bin/env python3
"""
generate_keys.py -- generate every DESILO-FHE key THOR needs and store it.

The Liberate key set under ./keys/keys0/ (pk, evk, gk, conjk, rotk_dict/...)
cannot be reused: those are Liberate DataStruct pickles and desilofhe has no
reader for them. This script writes a fresh desilofhe key set to a separate
directory so the two never mix.

Output layout (./keys_desilo/default/, or ./keys_desilo/compact/ with --compact):

    secret_key
    public_key
    relinearization_key
    conjugation_key
    bootstrap_key
    fixed_rotation/<delta>_<level>      one per rotation context
    params.json                         engine parameters these keys belong to

Run it once per mode before THOR_baseline.py:

    poetry run python generate_keys.py
    poetry run python generate_keys.py --compact
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from desilofhe import Engine

# ----------------------------------------------------------------------
# Copied verbatim from thor/he.py HE.__init__ so this script stays runnable
# on its own. If Desilo ever edits these tables upstream, re-copy them.
# ----------------------------------------------------------------------
# fmt: off
BOOTSTRAP_DELTAS_MEDIUM = set([  # compact mode
    1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 64, 96, 128, 160, 192, 224, 256, 512, 768, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 16384, 24576, 31744, 32000, 32256, 32512, 32736, 32744, 32752, 32760  # noqa: E501
])
BOOTSTRAP_DELTAS_LARGE = set([   # default mode
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 31744, 32256, 32736, 32752  # noqa: E501
])

ROTATION_CONTEXTS = [
    (0, 9), (1, 14), (2, 14), (3, 12), (4, 14), (5, 12), (6, 11), (8, 7), (16, 10), (32, 10), (64, 10), (128, 10), (240, 9), (256, 10), (496, 9), (512, 10), (752, 9), (1008, 9), (1024, 10), (1264, 9), (1520, 9), (1776, 9), (2032, 9), (2048, 8), (2272, 9), (2528, 9), (2784, 9), (3040, 9), (3184, 9), (3296, 9), (3312, 9), (3440, 9), (3552, 9), (3568, 9), (3696, 9), (3808, 9), (3824, 9), (3952, 9), (4064, 9), (4080, 9), (4304, 9), (4560, 9), (4816, 9), (5072, 9), (5328, 9), (5584, 9), (5840, 9), (6096, 9), (6336, 9), (6592, 9), (6848, 9), (7104, 9), (7264, 9), (7360, 9), (7392, 9), (7520, 9), (7616, 9), (7648, 9), (7776, 9), (7872, 9), (7904, 9), (8032, 9), (8128, 9), (8160, 9), (8368, 9), (8624, 9), (8880, 9), (9136, 9), (9392, 9), (9648, 9), (9904, 9), (10160, 9), (10400, 9), (10656, 9), (10912, 9), (11168, 9), (11344, 9), (11424, 9), (11472, 9), (11600, 9), (11680, 9), (11728, 9), (11856, 9), (11936, 9), (11984, 9), (12112, 9), (12192, 9), (12240, 9), (12432, 9), (12688, 9), (12944, 9), (13200, 9), (13456, 9), (13712, 9), (13968, 9), (14224, 9), (14464, 9), (14720, 9), (14976, 9), (15232, 9), (15424, 9), (15488, 9), (15552, 9), (15680, 9), (15744, 9), (15808, 9), (15936, 9), (16000, 9), (16064, 9), (16192, 9), (16256, 9), (16320, 9), (16384, 13), (16496, 9), (16752, 9), (17008, 9), (17264, 9), (17520, 9), (17776, 9), (18032, 9), (18288, 9), (18528, 9), (18784, 9), (19040, 9), (19296, 9), (19504, 9), (19552, 9), (19632, 9), (19760, 9), (19808, 9), (19888, 9), (20016, 9), (20064, 9), (20144, 9), (20272, 9), (20320, 9), (20400, 9), (20560, 9), (20816, 9), (21072, 9), (21328, 9), (21584, 9), (21840, 9), (22096, 9), (22352, 9), (22592, 9), (22848, 9), (23104, 9), (23360, 9), (23584, 9), (23616, 9), (23712, 9), (23840, 9), (23872, 9), (23968, 9), (24096, 9), (24128, 9), (24224, 9), (24352, 9), (24384, 9), (24480, 9), (24576, 13), (24624, 9), (24880, 9), (25136, 9), (25392, 9), (25648, 9), (25904, 9), (26160, 9), (26416, 9), (26656, 9), (26912, 9), (27168, 9), (27424, 9), (27664, 9), (27680, 9), (27792, 9), (27920, 9), (27936, 9), (28048, 9), (28176, 9), (28192, 9), (28304, 9), (28432, 9), (28448, 9), (28560, 9), (28672, 14), (28688, 9), (28944, 9), (29200, 9), (29456, 9), (29712, 9), (29968, 9), (30224, 9), (30480, 9), (30720, 14), (30976, 9), (31232, 9), (31488, 9), (31744, 13), (31872, 9), (32000, 9), (32128, 9), (32256, 13), (32384, 9), (32512, 13), (32640, 13), (32704, 13), (32736, 13), (32752, 13), (32757, 12), (32758, 12), (32759, 12), (32760, 12), (32761, 12), (32762, 11), (32763, 8), (32764, 13), (32765, 8), (32766, 13), (32767, 13)  # noqa: E501
]
# fmt: on


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate and store the DESILO-FHE keys used by THOR."
    )
    p.add_argument("--keys-dir", default=None,
                   help="Where to write. Default ./keys_desilo/default, "
                        "or ./keys_desilo/compact with --compact.")
    p.add_argument("--device", type=int, default=0, help="CUDA device index.")
    p.add_argument("--compact", action="store_true",
                   help="Generate for compact mode. Must match the flag you "
                        "pass to THOR_baseline.py / forward.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing key directory.")
    return p.parse_args()


def main():
    args = parse_args()
    key_size = "medium" if args.compact else "large"
    bootstrap_deltas = BOOTSTRAP_DELTAS_MEDIUM if args.compact else BOOTSTRAP_DELTAS_LARGE

    keydir = Path(args.keys_dir) if args.keys_dir else \
        Path("./keys_desilo") / ("compact" if args.compact else "default")

    if keydir.exists() and any(keydir.iterdir()):
        if not args.overwrite:
            print(f"ERROR: {keydir} exists and is not empty. Pass --overwrite.")
            return 1
        shutil.rmtree(keydir)
    (keydir / "fixed_rotation").mkdir(parents=True, exist_ok=True)

    print(f"Generating THOR keys into {keydir.resolve()}")
    print(f"  mode          : {'compact' if args.compact else 'default'}")
    print(f"  bootstrap key : {key_size}")
    print(f"  device        : cuda:{args.device}")
    print()

    started = time.perf_counter()

    # Engine parameters must match HE.__init__ exactly, or the keys will not
    # fit the engine that later loads them.
    engine = Engine(
        use_bootstrap_to_14_levels=True,
        mode="async gpu",
        device_id=args.device,
        compact=args.compact,
    )

    sizes = {}

    def emit(label, key, writer, path):
        t0 = time.perf_counter()
        writer(key, str(path))
        write_seconds = time.perf_counter() - t0
        resident = getattr(key, "nbytes", 0) / 1024**2
        on_disk = path.stat().st_size / 1024**2
        sizes[label] = dict(resident_mib=round(resident, 1), on_disk_mib=round(on_disk, 1))
        print(f"  {label:22s} resident {resident:9.1f} MiB | "
              f"on disk {on_disk:9.1f} MiB | {write_seconds:6.2f}s")

    secret_key = engine.create_secret_key()
    emit("secret_key", secret_key, engine.write_secret_key, keydir / "secret_key")

    # THOR encrypts symmetrically with the secret key and never uses a public
    # key, but it is part of a complete key set and costs little.
    public_key = engine.create_public_key(secret_key)
    emit("public_key", public_key, engine.write_public_key, keydir / "public_key")

    conjugation_key = engine.create_conjugation_key(secret_key)
    emit("conjugation_key", conjugation_key, engine.write_conjugation_key,
         keydir / "conjugation_key")

    relinearization_key = engine.create_relinearization_key(secret_key)
    emit("relinearization_key", relinearization_key, engine.write_relinearization_key,
         keydir / "relinearization_key")

    bootstrap_key = engine.create_bootstrap_key(secret_key, size=key_size)
    emit("bootstrap_key", bootstrap_key, engine.write_bootstrap_key,
         keydir / "bootstrap_key")

    # General rotation key: rotates by ANY delta with a single key
    # (engine.rotate(ct, rotation_key, delta)).
    #
    # THOR normally rotates the 52 bootstrap deltas using the BOOTSTRAP key,
    # which desilofhe also accepts as a rotation key -- fine on one machine,
    # but it pins the largest object in the system to whoever does the
    # rotating. For the NDP split the host needs those rotations without the
    # bootstrap key, and this is the small way to get them.
    #
    # Compare the two "resident" numbers printed below: that difference is the
    # data the NDP target no longer has to ship to the host.
    t0 = time.perf_counter()
    rotation_key = engine.create_rotation_key(secret_key)
    print(f"  {'rotation_key':22s} (created in {time.perf_counter() - t0:.1f}s)")
    emit("rotation_key", rotation_key, engine.write_rotation_key,
         keydir / "rotation_key")

    # Fixed rotation keys. Deltas covered by the bootstrap key are skipped,
    # exactly as HE.__init__ does -- generating them would waste time and disk.
    print()
    rotation_keys = []
    t0 = time.perf_counter()
    for delta, level in ROTATION_CONTEXTS:
        if delta == 0 or delta in bootstrap_deltas:
            continue
        key = engine.create_fixed_rotation_key(secret_key, delta, level=level)
        engine.write_fixed_rotation_key(key, str(keydir / "fixed_rotation" / f"{delta}_{level}"))
        rotation_keys.append((delta, level))
        if len(rotation_keys) % 25 == 0:
            print(f"  fixed rotation keys    {len(rotation_keys):4d} written "
                  f"({time.perf_counter() - t0:7.1f}s)")
    print(f"  fixed rotation keys    {len(rotation_keys):4d} total   "
          f"({time.perf_counter() - t0:7.1f}s)")

    # Records which engine these keys belong to. desilofhe's read_*_key does not
    # check that a key matches the loading engine -- a mismatched key loads
    # without error and yields wrong plaintexts -- so the loader compares this.
    (keydir / "params.json").write_text(json.dumps(dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        compact=bool(args.compact),
        bootstrap_key_size=key_size,
        use_bootstrap_to_14_levels=True,
        slot_count=int(engine.slot_count),
        max_level=int(engine.max_level),
        build_hash=str(engine.build_hash),
        rotation_key_count=len(rotation_keys),
        rotation_keys=[list(r) for r in rotation_keys],
        key_sizes=sizes,
    ), indent=2) + "\n")

    total_bytes = sum(f.stat().st_size for f in keydir.rglob("*") if f.is_file())
    print()
    bs = sizes.get("bootstrap_key", {}).get("resident_mib", 0)
    rot = sizes.get("rotation_key", {}).get("resident_mib", 0)
    if bs and rot:
        print("Host-side key options for the bootstrap deltas:")
        print(f"  bootstrap_key : {bs / 1024:8.2f} GiB resident  (what THOR uses by default)")
        print(f"  rotation_key  : {rot / 1024:8.2f} GiB resident  (equivalent for rotation only)")
        print(f"  difference    : {(bs - rot) / 1024:8.2f} GiB "
              f"({bs / rot:.1f}x smaller)")
        print()
    print(f"Done in {time.perf_counter() - started:.1f}s")
    print(f"  {5 + len(rotation_keys)} keys, {total_bytes / 1024**3:.2f} GiB "
          f"in {keydir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())