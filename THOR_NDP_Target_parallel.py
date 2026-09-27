#!/usr/bin/env python3
"""
THOR_NDP_Target_parallel.py -- bootstrap service with a pool of worker processes.

One command starts everything. The launcher spawns N workers and supervises
them; you do not start anything by hand in separate shells.

    launcher (this process)     owns the transport, holds NO Engine
      |- worker 0               own Engine + own key copies
      |- worker 1
      |- ...

Why processes and not threads: desilofhe's Engine is not safe for concurrent
calls on one instance (six threads on a shared engine segfaulted; the same
workload run serially was fine). Threads would therefore need a lock, which
would serialize bootstraps and defeat the purpose. Separate processes each get
their own Engine, and the keys are re-read per worker -- affordable here because
this is DRAM, not VRAM.

Why "spawn" and never fork: a parallel-mode engine has worker threads and a GPU
engine has CUDA state; fork() copies neither correctly. The launcher therefore
never builds an Engine before spawning.

Sizing for a 128 GB target: each worker holds relinearization + conjugation +
bootstrap keys, about 17.5 GiB (~12.5 GiB with --compact). Four workers is
roughly 70 GiB, which leaves headroom. --workers defaults to 4 for that reason.

    poetry run python THOR_NDP_Target_parallel.py --transport tcp --workers 4
    poetry run python THOR_NDP_Target_parallel.py --transport rdma --workers 4 \
        --mode parallel --threads 16
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
import multiprocessing as mp
import queue
import socket
import threading
import time
from pathlib import Path

from thor.ndp_parallel import PHEADER, PMAGIC, pframe, punframe, recv_exact

GPU_MODES = ("gpu", "async gpu")


def normalize_mode(mode: str) -> str:
    return {"async_gpu": "async gpu", "asyncgpu": "async gpu"}.get(mode, mode)


def parse_args():
    p = argparse.ArgumentParser(description="THOR NDP bootstrap target (worker pool).")
    p.add_argument("--workers", type=int, default=4,
                   help="Worker processes. Each holds a full key set (~17.5 GiB, "
                        "~12.5 GiB with --compact), so 4 fits a 128 GB target.")
    p.add_argument("--keys-dir", default=None,
                   help="Key set from generate_keys.py. MUST match the host's.")
    p.add_argument("--compact", action="store_true", help="Must match the host.")
    p.add_argument("--mode", default="parallel",
                   choices=["cpu", "parallel", "gpu", "async gpu", "async_gpu", "asyncgpu"],
                   help="Engine mode for each worker.")
    p.add_argument("--threads", type=int, default=None,
                   help="Thread count per worker for --mode parallel. With a pool, "
                        "size this to cores/workers -- one bootstrap stops scaling "
                        "well before it uses a whole large machine.")
    p.add_argument("--device", type=int, default=0, help="CUDA device for GPU modes.")
    p.add_argument("--transport", default="tcp", choices=["tcp", "rdma"])
    p.add_argument("--bind", default="0.0.0.0", help="TCP only.")
    p.add_argument("--port", type=int, default=9998, help="TCP only.")
    p.add_argument("--target-ip", default="192.168.100.1", help="RDMA only.")
    p.add_argument("--service", default="9999", help="RDMA only.")
    p.add_argument("--uds", default="/tmp/rdma_metadata.sock", help="RDMA only.")
    return p.parse_args()


# ======================================================================
# Worker process
# ======================================================================
def worker_main(wid, cfg, req_q, res_q, ready_q):
    """Own Engine, own keys, pull work forever."""
    from desilofhe import Engine

    mode = cfg["mode"]
    kwargs = dict(use_bootstrap_to_14_levels=True, mode=mode, compact=cfg["compact"])
    if mode in GPU_MODES:
        kwargs["device_id"] = cfg["device"]
    elif mode == "parallel" and cfg["threads"]:
        kwargs["thread_count"] = cfg["threads"]

    t0 = time.perf_counter()
    engine = Engine(**kwargs)
    keys_dir = Path(cfg["keys_dir"])
    relin = engine.read_relinearization_key(str(keys_dir / "relinearization_key"))
    conj = engine.read_conjugation_key(str(keys_dir / "conjugation_key"))
    bskey = engine.read_bootstrap_key(str(keys_dir / "bootstrap_key"))
    ready_q.put((wid, os.getpid(), round(time.perf_counter() - t0, 1)))

    served = 0
    while True:
        item = req_q.get()
        if item is None:
            break
        req_id, payload = item
        try:
            t0 = time.perf_counter()
            ct = engine.deserialize_ciphertext(payload)
            level_in = ct.level
            out = engine.bootstrap(ct, relin, conj, bskey)
            reply = bytes(engine.serialize_ciphertext(out))
            dt = time.perf_counter() - t0
            served += 1
            print(f"[w{wid}] req {req_id} level {level_in} -> {out.level} in {dt:.1f}s "
                  f"(served {served})", flush=True)
            res_q.put((req_id, reply, None))
        except Exception as exc:
            res_q.put((req_id, None, f"{type(exc).__name__}: {exc}"))


# ======================================================================
# Launcher
# ======================================================================
class Pool:
    def __init__(self, args, keys_dir):
        self.args = args
        mp.set_start_method("spawn", force=True)
        self.req_q, self.res_q, ready_q = mp.Queue(), mp.Queue(), mp.Queue()
        cfg = dict(mode=normalize_mode(args.mode), compact=args.compact,
                   threads=args.threads, device=args.device, keys_dir=str(keys_dir))

        print(f"Spawning {args.workers} workers (mode={cfg['mode']!r}"
              f"{', threads=' + str(args.threads) if args.threads and cfg['mode'] == 'parallel' else ''})")
        print(f"  each reads its own key set from {keys_dir}; "
              f"expect this to take a while and a lot of disk read")
        self.procs = [
            mp.Process(target=worker_main, args=(w, cfg, self.req_q, self.res_q, ready_q),
                       daemon=True)
            for w in range(args.workers)
        ]
        for p in self.procs:
            p.start()

        # Do not serve until every worker has its keys, or the first requests
        # queue behind a worker still reading GiBs off disk.
        for _ in range(args.workers):
            wid, pid, secs = ready_q.get()
            print(f"  worker {wid} ready (pid {pid}, {secs}s)")
        print("All workers ready.")

    def submit(self, req_id, payload):
        self.req_q.put((req_id, payload))

    def shutdown(self):
        for _ in self.procs:
            self.req_q.put(None)
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def serve_tcp(args, pool, stats):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"Target ready on tcp://{args.bind}:{args.port}")

    while True:
        conn, peer = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Host connected from {peer}")
        send_lock = threading.Lock()
        stop = threading.Event()

        def sender():
            # Replies leave in completion order, not arrival order -- the id in
            # the frame is what pairs them up on the host.
            while not stop.is_set():
                try:
                    req_id, reply, err = pool.res_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if err:
                    print(f"  worker error on req {req_id}: {err}")
                    continue
                with send_lock:
                    try:
                        conn.sendall(pframe(req_id, reply))
                    except OSError:
                        return
                stats["served"] += 1

        t = threading.Thread(target=sender, daemon=True)
        t.start()
        try:
            while True:
                head = recv_exact(conn, PHEADER.size)
                if head is None:
                    break
                magic, req_id, length = PHEADER.unpack(head)
                if magic != PMAGIC:
                    raise ValueError(f"bad magic {magic!r} -- parallel host required")
                payload = recv_exact(conn, length)
                if payload is None:
                    break
                stats["received"] += 1
                pool.submit(req_id, payload)
        except Exception as exc:
            print(f"Connection error: {type(exc).__name__}: {exc}")
        finally:
            stop.set()
            t.join(timeout=2)
            conn.close()
            print("Host disconnected; waiting for the next connection.")


def serve_rdma(args, pool, stats):
    """UNTESTED on hardware -- but structured to avoid the RDMA pitfalls below.

    ONE thread owns the CMID. Nothing else touches it.

    An earlier version had the UDS handler and a sender thread both posting to
    the queue pair and both draining completions. post_read and post_send share
    the same send completion queue, so the two threads could reap each other's
    completions; and the `if get_send_comp() is None` path continued without
    reaping, leaking a send-queue slot and closing an MR the HCA might still be
    reading. Slots leak, the queue fills, and post_send returns ENOMEM.

    So: a single loop alternates between draining UDS offload events and
    draining finished bootstraps, performing exactly one RDMA operation at a
    time and reaping its completion before starting the next. Worker processes
    still run concurrently -- only the wire is serialized, and at a few MiB per
    transfer it is nowhere near the bottleneck against a ~14s bootstrap.

    Buffers are registered ONCE and reused. Registering an MR per operation is
    expensive and pins memory that is only released when the MR is closed.
    """
    import ctypes
    import selectors
    import struct as _struct

    from pyverbs.cmid import CMID, AddrInfo
    from pyverbs.librdmacm_enums import RAI_PASSIVE, rdma_port_space
    from pyverbs.qp import QPCap, QPInitAttr

    # Same caps the Liberate target used on this hardware. Do not raise these
    # without checking `ibv_devinfo -v | grep max_qp_wr` -- an unsupported value
    # is silently clamped, and you end up with a smaller queue than you think.
    cap = QPCap(max_send_wr=16, max_recv_wr=16, max_send_sge=8)
    cai = AddrInfo(src=args.target_ip, src_service=args.service,
                   port_space=rdma_port_space.RDMA_PS_TCP, flags=RAI_PASSIVE)
    cid = CMID(creator=cai, qp_init_attr=QPInitAttr(cap=cap))
    print("RDMA listening")
    cid.listen()
    conn_id = cid.get_request()
    conn_id.accept()
    print("RDMA connected")

    # One MR per transfer, sized exactly to the message -- this is what the
    # Liberate target did on this hardware. A single oversized MR reused for
    # every send looked tidier but post_send then failed with ENOMEM on the
    # first reply, so the transfer length is evidently taken from the MR rather
    # than from the size argument. Register exactly what you intend to move.
    def reap_send(what):
        """Block until this operation's completion is in hand, and check it.

        Never skip this: an unreaped completion holds its send-queue slot
        forever, and post_send then fails with ENOMEM. A non-success status
        also puts the queue pair into ERROR, after which every later post fails
        the same way -- so surface it here rather than three operations later.
        """
        wc = conn_id.get_send_comp()
        if wc is None:
            raise RuntimeError(f"no completion for {what}")
        status = getattr(wc, "status", None)
        if status not in (None, 0):
            raise RuntimeError(f"{what} completed with status {status} "
                               f"(non-zero puts the QP in ERROR; every "
                               f"subsequent post will fail)")
        return wc

    if os.path.exists(args.uds):
        os.remove(args.uds)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.uds)
    srv.listen(64)
    srv.setblocking(False)
    sel = selectors.DefaultSelector()
    sel.register(srv, selectors.EVENT_READ)

    pending = []          # UDS connections with metadata ready
    # req_id -> (uds_conn, local_mr). The working single-stream target keeps BOTH
    # of these open until after the reply has been sent, and closes them in that
    # order. Closing them early is the difference that broke this path: the UDS
    # connection is how the NDP driver delivered the metadata, and tearing it
    # down mid-transfer takes driver state with it, after which post_send fails
    # with ENOMEM. The bootstrap is asynchronous here, so the resources have to
    # be parked until the worker comes back rather than held on the stack.
    inflight = {}

    def drain_uds(timeout):
        for key, _ in sel.select(timeout=timeout):
            if key.fileobj is srv:
                conn, _ = srv.accept()
                conn.setblocking(False)
                sel.register(conn, selectors.EVENT_READ)
            else:
                pending.append(key.fileobj)
                sel.unregister(key.fileobj)

    print(f"Target ready. Waiting for offload events on {args.uds}")
    fmt = "<QQII50s"
    while True:
        drain_uds(0.05)

        # 1. pull in any newly announced host buffers
        while pending:
            conn = pending.pop()
            data = conn.recv(128)
            if not data:
                conn.close()
                continue
            rkey, addr, length, name_len, name = _struct.unpack(
                fmt, data[:_struct.calcsize(fmt)])
            local_mr = conn_id.reg_msgs(length)
            conn_id.post_read(local_mr, length, addr, rkey)
            reap_send("READ")
            view = bytes((ctypes.c_uint8 * length).from_address(local_mr.buf))
            req_id, payload = punframe(view)
            # hold conn and local_mr open until the reply goes out
            inflight[req_id] = (conn, local_mr)
            stats["received"] += 1
            pool.submit(req_id, payload)

        # 2. ship any finished bootstraps back
        while True:
            try:
                req_id, reply, err = pool.res_q.get_nowait()
            except queue.Empty:
                break
            if err:
                print(f"  worker error on req {req_id}: {err}")
                continue
            blob = pframe(req_id, reply)
            out_mr = conn_id.reg_msgs(len(blob))
            (ctypes.c_uint8 * len(blob)).from_address(out_mr.buf)[:] = blob
            # NOTE: pyverbs signature is post_send(mr, flags, length) -- the
            # second POSITIONAL argument is FLAGS, not length. Passing the size
            # there sets whatever IBV_SEND_* bits happen to be in the number;
            # any size with bit 3 set turns on IBV_SEND_INLINE, and inlining a
            # multi-MiB message makes the provider return ENOMEM. Always pass
            # length as a keyword.
            conn_id.post_send(out_mr, length=len(blob))
            reap_send(f"SEND req {req_id}")

            # Same teardown order as the working single-stream target:
            # local_mr, then out_mr, then the UDS connection -- all after the
            # send has completed, never before.
            conn, local_mr = inflight.pop(req_id, (None, None))
            if local_mr is not None:
                local_mr.close()
            out_mr.close()
            if conn is not None:
                conn.close()

            stats["served"] += 1
            print(f"  sent reply for req {req_id} ({len(blob) / 1024**2:.1f} MiB)",
                  flush=True)


def main():
    args = parse_args()
    keys_dir = Path(args.keys_dir) if args.keys_dir else \
        Path("./keys_desilo") / ("compact" if args.compact else "default")
    params = json.loads((keys_dir / "params.json").read_text())
    if bool(params.get("compact")) != args.compact:
        raise SystemExit(f"{keys_dir} built with compact={params.get('compact')}, "
                         f"target running compact={args.compact}.")

    started = time.perf_counter()
    pool = Pool(args, keys_dir)
    stats = {"received": 0, "served": 0}
    try:
        (serve_tcp if args.transport == "tcp" else serve_rdma)(args, pool, stats)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        elapsed = time.perf_counter() - started
        print(f"Received {stats['received']}, served {stats['served']} "
              f"bootstraps in {elapsed:.0f}s "
              f"({stats['served'] / elapsed * 3600:.0f}/hour aggregate)")
        pool.shutdown()


if __name__ == "__main__":
    main()