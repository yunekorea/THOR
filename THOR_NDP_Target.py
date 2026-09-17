#!/usr/bin/env python3
"""
THOR_NDP_target.py -- bootstrap service running on a CPU desilofhe engine.

Ported from the Liberate THOR_NDP_target.py. Three things changed:

1. The engine mode is selectable with --mode. Desilo offers four: "cpu"
   (single-threaded) and "parallel" (multi-threaded, 4 threads by default,
   settable via thread_count) run without a GPU; "gpu" and "async gpu" use
   CUDA and take a device_id. Default is "parallel" -- CPU bootstrapping is
   slow enough that threads matter. All four build the same ring as the host:
   Engine(use_bootstrap_to_14_levels=True) gives slot_count=32768 and
   max_level=14 regardless of mode (confirmed on desilofhe 1.16), so the same
   key set loads into any of them.

2. Keys are read, not loaded-and-paged. The LRU/Belady rotation-key caches from
   the Liberate target are gone: bootstrap deltas are fused into one opaque
   BootstrapKey object here, so there is no per-delta dict to intercept.
   The target reads exactly three keys -- relinearization, conjugation,
   bootstrap. It never sees the secret key, which is a nice property for the
   threat model: the target bootstraps ciphertexts it cannot decrypt.

3. The payload is desilofhe's own ciphertext serialization instead of a
   hand-rolled DataStruct walk, so this file no longer needs to know anything
   about tensor layout.

    poetry run python THOR_NDP_target.py --transport tcp --mode parallel --threads 64
    poetry run python THOR_NDP_target.py --transport rdma --mode gpu --device 0
    poetry run python THOR_NDP_target.py --transport rdma --mode async_gpu
"""

import argparse
import json
import socket
import time
from pathlib import Path

from desilofhe import Engine

from thor.he_ndp import MAX_CT_BYTES, frame, unframe


def parse_args():
    p = argparse.ArgumentParser(description="THOR NDP bootstrap target.")
    p.add_argument("--keys-dir", default=None,
                   help="Key set from generate_keys.py. MUST be the same set the "
                        "host uses, or bootstrapping returns garbage.")
    p.add_argument("--compact", action="store_true",
                   help="Must match the host.")
    p.add_argument("--mode", default="parallel",
                   choices=["cpu", "parallel", "gpu", "async gpu",
                            "async_gpu", "asyncgpu"],
                   help="Where bootstrapping runs. 'cpu' single-threaded CPU; "
                        "'parallel' multi-threaded CPU (default); 'gpu' CUDA; "
                        "'async gpu' CUDA with asynchronous execution. "
                        "'async_gpu' and 'asyncgpu' are accepted spellings of "
                        "'async gpu' so you need not quote the space.")
    p.add_argument("--threads", type=int, default=None,
                   help="Thread count for --mode parallel (library default is 4). "
                        "Ignored by the GPU modes.")
    p.add_argument("--device", type=int, default=0,
                   help="CUDA device index for the GPU modes. Ignored on CPU.")
    p.add_argument("--transport", default="tcp", choices=["tcp", "rdma"])
    p.add_argument("--bind", default="0.0.0.0", help="TCP transport only.")
    p.add_argument("--port", type=int, default=9998, help="TCP transport only.")
    p.add_argument("--host-ip", default="192.168.100.2", help="RDMA transport only.")
    p.add_argument("--target-ip", default="192.168.100.1", help="RDMA transport only.")
    p.add_argument("--service", default="9999", help="RDMA transport only.")
    p.add_argument("--uds", default="/tmp/rdma_metadata.sock",
                   help="Unix socket the NDP driver writes offload metadata to.")
    return p.parse_args()


# ======================================================================
# Engine and keys
# ======================================================================
GPU_MODES = ("gpu", "async gpu")


def normalize_mode(mode: str) -> str:
    """Accept async_gpu / asyncgpu for the space-containing 'async gpu'."""
    return {"async_gpu": "async gpu", "asyncgpu": "async gpu"}.get(mode, mode)


def engine_init(args):
    mode = normalize_mode(args.mode)
    kwargs = dict(use_bootstrap_to_14_levels=True, mode=mode, compact=args.compact)

    detail = ""
    if mode in GPU_MODES:
        # Engine takes device_id only for the CUDA modes.
        kwargs["device_id"] = args.device
        detail = f", device_id={args.device}"
        if args.threads:
            print(f"  (ignoring --threads {args.threads}: it applies to 'parallel' only)")
    elif mode == "parallel" and args.threads:
        kwargs["thread_count"] = args.threads
        detail = f", threads={args.threads}"

    print(f"Engine init (mode={mode!r}{detail}): ", end="", flush=True)
    t0 = time.perf_counter()
    engine = Engine(**kwargs)
    print(f"DONE ({time.perf_counter() - t0:.1f}s, "
          f"slot_count={engine.slot_count}, max_level={engine.max_level})")

    if mode in GPU_MODES:
        # Worth saying out loud: with the target bootstrapping on a GPU, the
        # split stops being "GPU host offloads to CPU storage node" and becomes
        # a two-GPU split. Useful for isolating transport cost from the CPU
        # bootstrap penalty -- run the same workload under 'gpu' and under
        # 'parallel' and the difference is the device, not the wire.
        print("  NOTE: target is bootstrapping on a GPU, not the CPU. Key residency "
              "moves to VRAM: the bootstrap key alone is ~17 GiB (~12 GiB compact).")
    return engine


def key_init(engine, keys_dir: Path, compact: bool):
    params = json.loads((keys_dir / "params.json").read_text())
    if bool(params.get("compact")) != compact:
        raise SystemExit(f"{keys_dir} built with compact={params.get('compact')}, "
                         f"target running compact={compact}.")
    if params.get("slot_count") != int(engine.slot_count):
        raise SystemExit(
            f"ring mismatch: keys slot_count={params.get('slot_count')}, "
            f"CPU engine slot_count={engine.slot_count}."
        )

    keys = {}
    for name, reader in (("relinearization_key", engine.read_relinearization_key),
                         ("conjugation_key", engine.read_conjugation_key),
                         ("bootstrap_key", engine.read_bootstrap_key)):
        print(f"  {name}: ", end="", flush=True)
        t0 = time.perf_counter()
        keys[name] = reader(str(keys_dir / name))
        print(f"DONE ({time.perf_counter() - t0:.1f}s, "
              f"{keys[name].nbytes / 1024**3:.2f} GiB resident)")
    # The secret key is deliberately not read: bootstrapping does not need it.
    return keys


def do_bootstrap(engine, keys, payload: bytes, stats) -> bytes:
    stats["calls"] += 1
    n = stats["calls"]

    t0 = time.perf_counter()
    ct = engine.deserialize_ciphertext(payload)
    deser = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = engine.bootstrap(ct, keys["relinearization_key"],
                           keys["conjugation_key"], keys["bootstrap_key"])
    bs = time.perf_counter() - t0

    t0 = time.perf_counter()
    reply = engine.serialize_ciphertext(out)
    ser = time.perf_counter() - t0

    stats["bootstrap_seconds"] += bs
    stats["codec_seconds"] += deser + ser
    print(f"[bs #{n:04d}] level {ct.level} -> {out.level} | "
          f"deserialize {deser:.2f}s | bootstrap {bs:.1f}s | serialize {ser:.2f}s")
    return bytes(reply)


# ======================================================================
# Transports
# ======================================================================
def serve_tcp(args, engine, keys, stats):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"Target ready on tcp://{args.bind}:{args.port}. Waiting for the host.")

    def recv_exact(sock, n):
        chunks, got = [], 0
        while got < n:
            b = sock.recv(min(1 << 20, n - got))
            if not b:
                return None
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)

    from thor.he_ndp import _HEADER, _MAGIC
    while True:
        conn, peer = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Host connected from {peer}")
        try:
            while True:
                head = recv_exact(conn, _HEADER.size)
                if head is None:
                    break
                magic, length = _HEADER.unpack(head)
                if magic != _MAGIC:
                    raise ValueError(f"bad magic {magic!r}")
                payload = recv_exact(conn, length)
                if payload is None:
                    break
                conn.sendall(frame(do_bootstrap(engine, keys, payload, stats)))
        except Exception as exc:
            print(f"Connection error: {type(exc).__name__}: {exc}")
        finally:
            conn.close()
            print("Host disconnected; waiting for the next connection.")


def serve_rdma(args, engine, keys, stats):
    """Original NDP path: UDS doorbell from the driver, RDMA read/send for data."""
    import ctypes
    import os
    import selectors
    import struct

    from pyverbs.cmid import CMID, AddrInfo
    from pyverbs.librdmacm_enums import RAI_PASSIVE, rdma_port_space
    from pyverbs.qp import QPCap, QPInitAttr

    cap = QPCap(max_send_wr=16, max_recv_wr=16, max_send_sge=8)
    cai = AddrInfo(src=args.target_ip, src_service=args.service,
                   port_space=rdma_port_space.RDMA_PS_TCP, flags=RAI_PASSIVE)
    cid = CMID(creator=cai, qp_init_attr=QPInitAttr(cap=cap))
    print("RDMA listening")
    cid.listen()
    conn_id = cid.get_request()
    conn_id.accept()
    print("RDMA connected")

    if os.path.exists(args.uds):
        os.remove(args.uds)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.uds)
    srv.listen(1)
    srv.setblocking(False)
    sel = selectors.DefaultSelector()

    def handle(conn):
        data = conn.recv(128)
        if data:
            fmt = "<QQII50s"
            rkey, addr, length, name_len, name = struct.unpack(fmt, data[:struct.calcsize(fmt)])
            local_mr = conn_id.reg_msgs(length)
            conn_id.post_read(local_mr, length, addr, rkey)
            if conn_id.get_send_comp() is None:
                raise RuntimeError("no READ completion")

            view = bytes((ctypes.c_uint8 * length).from_address(local_mr.buf))
            reply = frame(do_bootstrap(engine, keys, unframe(view), stats))

            out_mr = conn_id.reg_msgs(len(reply))
            (ctypes.c_uint8 * len(reply)).from_address(out_mr.buf)[:] = reply
            conn_id.post_send(out_mr, len(reply))
            if conn_id.get_send_comp() is None:
                raise RuntimeError("no SEND completion")
            local_mr.close()
            out_mr.close()
        sel.unregister(conn)
        conn.close()

    def accept(sock):
        conn, _ = sock.accept()
        conn.setblocking(False)
        sel.register(conn, selectors.EVENT_READ, data=lambda k: handle(k.fileobj))

    sel.register(srv, selectors.EVENT_READ, data=lambda k: accept(k.fileobj))
    print(f"Target ready. Waiting for offload events on {args.uds}")
    while True:
        for key, _ in sel.select():
            key.data(key)


def main():
    args = parse_args()
    keys_dir = Path(args.keys_dir) if args.keys_dir else \
        Path("./keys_desilo") / ("compact" if args.compact else "default")

    engine = engine_init(args)
    print(f"Loading keys from {keys_dir} (secret key is not read)")
    keys = key_init(engine, keys_dir, args.compact)

    stats = {"calls": 0, "bootstrap_seconds": 0.0, "codec_seconds": 0.0}
    try:
        (serve_tcp if args.transport == "tcp" else serve_rdma)(args, engine, keys, stats)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        n = stats["calls"]
        if n:
            print(f"Served {n} bootstraps | "
                  f"bootstrap {stats['bootstrap_seconds']:.1f}s "
                  f"({stats['bootstrap_seconds']/n:.1f}s each) | "
                  f"codec {stats['codec_seconds']:.1f}s")


if __name__ == "__main__":
    main()