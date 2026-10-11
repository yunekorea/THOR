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
import glob
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


# ======================================================================
# NUMA topology
# ======================================================================
# On a two-socket box each socket has its own memory controllers, so a worker
# reading a key that lives on the other socket pays the UPI hop for every cache
# line -- higher latency and a shared interconnect ceiling instead of a private
# set of channels. Spreading workers across nodes and keeping each one's keys
# local is the difference between using both sockets' channels and funnelling
# everything through one socket's.
def numa_topology():
    """[(node_id, [cpu ids])] from sysfs, or [] if not a NUMA machine."""
    out = []
    for path in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
        try:
            nid = int(path.rsplit("node", 1)[1])
            cpus = parse_cpulist(Path(f"{path}/cpulist").read_text().strip())
        except (OSError, ValueError):
            continue
        if cpus:
            out.append((nid, cpus))
    return sorted(out)


def format_cpulist(cpus):
    """[0..15, 32..47] -> '0-15,32-47'.

    Printing first-last would render that node as '0-47', which looks like the
    mask leaked onto the other socket. A node's CPUs are not contiguous once
    hyperthread siblings are numbered separately, so collapse to real ranges.
    """
    cpus = sorted(cpus)
    if not cpus:
        return "-"
    parts, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if c is not None:
            start = prev = c
    return ",".join(parts)


def parse_cpulist(spec):
    """'0-15,32-47' -> [0..15, 32..47]."""
    cpus = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def node_meminfo(nid):
    """(MemFree MiB, hugepages reserved MiB) for one node."""
    free = huge = 0
    try:
        for line in Path(f"/sys/devices/system/node/node{nid}/meminfo").read_text().splitlines():
            if "MemFree:" in line:
                free = int(line.split()[-2]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    # SPDK's hugepage reservation shows as 'used' to the kernel but appears in no
    # process's RSS, so it is invisible unless looked for explicitly -- and it is
    # exactly what makes a node unable to hold its workers' keys.
    for pat in ("hugepages-2048kB", "hugepages-1048576kB"):
        try:
            p = Path(f"/sys/devices/system/node/node{nid}/hugepages/{pat}/nr_hugepages")
            n = int(p.read_text().strip())
            huge += n * (2 if "2048" in pat else 1024)
        except (OSError, ValueError):
            pass
    return free, huge


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
    p.add_argument("--expect-bootstraps", type=int, default=0, metavar="N",
                   help="Exit automatically once N bootstraps have been served AND "
                        "their replies sent. 0 (the default) keeps the old "
                        "behaviour: serve until Ctrl-C. One forward pass is 294 "
                        "bootstraps, so a host running 4 parallel streams is "
                        "294*4=1176. NOTE this is how many bootstraps the HOST "
                        "will send; it is deliberately NOT derived from --workers, "
                        "because target workers and host streams are independent "
                        "knobs -- 4 workers serving a 2-stream host would sit "
                        "waiting forever for a 1176th request that never comes.")
    p.add_argument("--connect-after-init", action="store_true",
                   help="Restore the old startup order: spawn workers and read "
                        "keys first, open the transport afterwards. The default is "
                        "the reverse, because the host opens its transport before "
                        "doing its own init and would otherwise find nothing "
                        "listening.")
    p.add_argument("--numa-bind", action="store_true",
                   help="Pin each worker to one NUMA node, round-robin: worker 0 "
                        "to node 0, worker 1 to node 1, worker 2 back to node 0, "
                        "and so on. Each worker's keys then land on that node's "
                        "own DIMMs, so workers use both sockets' memory channels "
                        "instead of funnelling every access through one socket's. "
                        "Size --threads to cores-per-node divided by "
                        "workers-per-node.")
    p.add_argument("--numa-nodes", default=None, metavar="LIST",
                   help="Restrict --numa-bind to these nodes, e.g. '0,1' or '0'. "
                        "Default: every node the machine reports.")
    p.add_argument("--key-gib", type=float, default=17.5, metavar="GIB",
                   help="Resident key-set size per worker, for the per-node "
                        "capacity check (default 17.5; use ~12.5 with --compact).")
    p.add_argument("--bootstrap-log", default=None, metavar="CSV",
                   help="Append one row per bootstrap: wall-clock start and end, "
                        "levels, duration. Each worker writes its OWN file, "
                        "<CSV>.w<id>, so concurrent processes never interleave "
                        "into one file. Feed these plus membw.py's CSV to "
                        "membw_join.py to get DRAM bandwidth broken down by how "
                        "many workers were bootstrapping at the time.")
    return p.parse_args()


class QuotaMet(Exception):
    """Raised once --expect-bootstraps replies have gone out, to unwind the loop."""


def quota_met(args, stats) -> bool:
    return bool(args.expect_bootstraps) and stats["served"] >= args.expect_bootstraps


# ======================================================================
# Worker process
# ======================================================================
def worker_main(wid, cfg, req_q, res_q, ready_q):
    """Own Engine, own keys, pull work forever."""
    from desilofhe import Engine

    # Pin BEFORE the Engine exists. Two reasons, both load-bearing:
    #   1. desilofhe's parallel mode spawns its own threads, and threads inherit
    #      the affinity mask in place when they are created. Pinning afterwards
    #      would leave those threads free to roam across both sockets.
    #   2. Linux's default memory policy is first-touch-local, so once this
    #      process only runs on node N's CPUs, the ~17.5 GiB of keys it reads
    #      lands on node N's DIMMs automatically. No libnuma, no explicit
    #      membind, and -- unlike a strict `numactl --membind` -- no OOM kill if
    #      the node fills up; it degrades to remote pages instead of dying.
    node_info = ""
    if cfg.get("numa_map"):
        nid, cpus = cfg["numa_map"][wid % len(cfg["numa_map"])]
        try:
            os.sched_setaffinity(0, set(cpus))
            got = sorted(os.sched_getaffinity(0))
            node_info = (f" [NUMA node {nid}, {len(got)} cpus: "
                         f"{format_cpulist(got)}]")
        except OSError as exc:
            node_info = f" [NUMA bind FAILED: {exc}]"

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
    ready_q.put((wid, os.getpid(), round(time.perf_counter() - t0, 1), node_info))

    # Own file per worker. Four processes appending to one file would interleave
    # partial lines; separate files need no locking and cost nothing, and
    # membw_join.py globs them back together.
    bslog = None
    if cfg.get("bootstrap_log"):
        bslog = open(f"{cfg['bootstrap_log']}.w{wid}", "w", buffering=1)
        bslog.write("worker,seq,t_start,t_end,duration_s,level_in,level_out\n")

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
            # Bracket the bootstrap call ONLY. deserialize/serialize are left
            # outside: their memory traffic is not bootstrap traffic, and
            # including it would inflate the figure being measured.
            wall0 = time.time()
            out = engine.bootstrap(ct, relin, conj, bskey)
            wall1 = time.time()
            reply = bytes(engine.serialize_ciphertext(out))
            dt = time.perf_counter() - t0
            served += 1
            if bslog is not None:
                bslog.write(f"{wid},{served},{wall0:.3f},{wall1:.3f},"
                            f"{wall1 - wall0:.3f},{level_in},{out.level}\n")
            print(f"[w{wid}] req {req_id} level {level_in} -> {out.level} in {dt:.1f}s "
                  f"(served {served})", flush=True)
            res_q.put((req_id, reply, None))
        except Exception as exc:
            res_q.put((req_id, None, f"{type(exc).__name__}: {exc}"))

    if bslog is not None:
        bslog.close()


# ======================================================================
# Launcher
# ======================================================================
def plan_numa(args):
    """Decide the worker -> node mapping and sanity-check it. [] disables binding."""
    if not args.numa_bind:
        return []

    topo = numa_topology()
    if args.numa_nodes:
        wanted = {int(x) for x in args.numa_nodes.replace(" ", "").split(",") if x}
        topo = [(n, c) for n, c in topo if n in wanted]
        missing = wanted - {n for n, _ in topo}
        if missing:
            raise SystemExit(f"--numa-nodes names node(s) {sorted(missing)} that "
                             f"this machine does not report.")
    if not topo:
        print("  --numa-bind requested but no NUMA nodes found; running unpinned.")
        return []
    if len(topo) == 1:
        print(f"  --numa-bind: only one node ({topo[0][0]}); pinning is a no-op "
              f"for locality.")

    # In a GPU mode the key set lives in VRAM, not in this node's DIMMs, so the
    # host-RAM capacity check below would be measuring the wrong pool and would
    # warn about a shortage that does not apply. Pinning still has a small point
    # (staging buffers and the PCIe path stay on one node), so keep the binding
    # and only drop the memory arithmetic.
    on_gpu = normalize_mode(args.mode) in GPU_MODES
    if on_gpu:
        print(f"  (GPU mode: keys go to VRAM on device {args.device}, "
              f"~{args.workers * args.key_gib:.0f} GiB for {args.workers} "
              f"worker(s) -- host-RAM check below does not apply)")

    print(f"NUMA layout ({len(topo)} node{'s' if len(topo) != 1 else ''}):")
    per_node = {}
    for i in range(args.workers):
        nid = topo[i % len(topo)][0]
        per_node.setdefault(nid, []).append(i)

    need_warn = False
    for nid, cpus in topo:
        mine = per_node.get(nid, [])
        free_mib, huge_mib = node_meminfo(nid)
        need_gib = len(mine) * args.key_gib
        threads_each = args.threads or 4
        line = (f"  node {nid}: {len(cpus)} cpus, {free_mib / 1024:.0f} GiB free")
        if huge_mib:
            line += f" (+{huge_mib / 1024:.0f} GiB hugepages reserved)"
        line += f" <- workers {mine or '-'}"
        print(line)
        if mine and on_gpu:
            print(f"    keys in VRAM, not here")
        elif mine:
            print(f"    needs ~{need_gib:.0f} GiB of keys, "
                  f"{len(mine) * threads_each} engine threads on {len(cpus)} cpus")
            if free_mib / 1024 < need_gib:
                spill = ("pages will spill to another node and you lose the "
                         "locality you are\n       pinning for"
                         if len(topo) > 1 else
                         "the run will swap or be OOM-killed")
                print(f"    !! only {free_mib / 1024:.0f} GiB free but "
                      f"~{need_gib:.0f} GiB of keys wanted -- {spill}.")
                need_warn = True
            if len(mine) * threads_each > len(cpus):
                print(f"    !! {len(mine) * threads_each} engine threads on "
                      f"{len(cpus)} cpus oversubscribes this node; "
                      f"use --threads {max(1, len(cpus) // len(mine))}")
                need_warn = True

    if need_warn:
        print("\n  (SPDK hugepages count against a node's free memory but appear "
              "in no process's\n   RSS, so they are easy to miss -- they are "
              "included above.)")
    return [(n, c) for n, c in topo]


class Pool:
    def __init__(self, args, keys_dir):
        self.args = args
        mp.set_start_method("spawn", force=True)
        self.req_q, self.res_q, ready_q = mp.Queue(), mp.Queue(), mp.Queue()
        numa_map = plan_numa(args)
        cfg = dict(mode=normalize_mode(args.mode), compact=args.compact,
                   threads=args.threads, device=args.device, keys_dir=str(keys_dir),
                   bootstrap_log=args.bootstrap_log, numa_map=numa_map)

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
        #
        # Poll with a timeout rather than blocking: a worker that dies during
        # init -- CUDA out of memory is the usual way, when several workers each
        # want ~17.5 GiB on one GPU -- never reaches ready_q, and a bare get()
        # would wait for it forever with nothing printed. Check liveness instead
        # and say which worker died.
        ready = 0
        while ready < args.workers:
            try:
                wid, pid, secs, node_info = ready_q.get(timeout=5)
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(self.procs)
                        if not p.is_alive()]
                if dead:
                    for p in self.procs:
                        if p.is_alive():
                            p.terminate()
                    detail = ", ".join(f"worker {i} (exit {code})"
                                       for i, code in dead)
                    hint = ""
                    if cfg["mode"] in GPU_MODES:
                        hint = (f"\nEach worker needs ~{args.key_gib:.0f} GiB of "
                                f"VRAM for its own key set, so {args.workers} "
                                f"workers need ~{args.workers * args.key_gib:.0f} "
                                f"GiB on device {args.device}.\nCheck "
                                f"`nvidia-smi`; if that does not fit, use "
                                f"--workers 1 or --compact. Note that separate "
                                f"CUDA contexts on one device time-slice rather "
                                f"than run\nconcurrently, so extra workers on a "
                                f"single GPU cost VRAM without adding "
                                f"throughput.")
                    raise SystemExit(
                        f"\nWorker died during init: {detail}.{hint}\n"
                        f"The traceback is above this line."
                    )
                continue
            ready += 1
            print(f"  worker {wid} ready (pid {pid}, {secs}s){node_info}")
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


# ======================================================================
# Transport setup -- opened before the workers spawn
# ======================================================================
# The host opens its transport as the FIRST thing it does, before building its
# own engine and reading its own keys. Spawning four workers that each read a
# ~17 GiB key set takes minutes; if the endpoint only appears afterwards, the
# host is refused and dies. So the endpoint is opened up front and the serve
# loops are handed the open handle.
#
# Requests that arrive before the pool is ready are parked, not dropped: TCP
# payloads sit in the kernel receive buffer, and RDMA doorbells sit in the UDS
# backlog with the host's buffer still registered (the target initiates the
# READ, so nothing moves until we move it).
#
# Opening before mp.Process.start() is safe because the start method is "spawn":
# children are fresh interpreters that inherit only the queue pipes, so no
# child ever sees the CMID or the listening socket.
def transport_open(args):
    if args.transport == "tcp":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.bind, args.port))
        srv.listen(1)
        print(f"Listening on tcp://{args.bind}:{args.port} "
              f"(host may connect now; served once workers are ready)")
        return {"srv": srv}

    import selectors

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
    cid.listen()
    print(f"RDMA listening on {args.target_ip}:{args.service} -- waiting for host")
    conn_id = cid.get_request()
    conn_id.accept()
    print("RDMA connected")

    if os.path.exists(args.uds):
        os.remove(args.uds)
    uds = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    uds.bind(args.uds)
    uds.listen(64)
    uds.setblocking(False)
    sel = selectors.DefaultSelector()
    sel.register(uds, selectors.EVENT_READ)
    print(f"Offload doorbell open at {args.uds}")
    return {"cid": cid, "conn_id": conn_id, "uds": uds, "sel": sel}


def serve_tcp(args, link, pool, stats):
    srv = link["srv"]
    print(f"Target ready on tcp://{args.bind}:{args.port}")

    while True:
        conn, peer = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Host connected from {peer}")
        send_lock = threading.Lock()
        stop = threading.Event()
        quota = threading.Event()

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

                # The quota is reached here, in the sender, but the reader
                # thread is parked in a blocking recv_exact and will not notice
                # a flag. Half-closing the read side makes its recv return b""
                # so recv_exact yields None and the reader falls out of its
                # loop; quota then tells the outer loop this was deliberate
                # rather than the host hanging up.
                if quota_met(args, stats):
                    quota.set()
                    stop.set()
                    try:
                        conn.shutdown(socket.SHUT_RD)
                    except OSError:
                        pass
                    return

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
        if quota.is_set():
            raise QuotaMet
        print("Host disconnected; waiting for the next connection.")


def serve_rdma(args, link, pool, stats):
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

    conn_id = link["conn_id"]
    srv = link["uds"]
    sel = link["sel"]

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

    # The UDS listener and its selector were created by transport_open(), before
    # the workers spawned -- do NOT rebind them here. Rebinding would unlink the
    # socket the NDP driver may already have queued a doorbell on.

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

            # Checked here, after the send completed and this request's MRs and
            # UDS handle are released -- never mid-transfer.
            if quota_met(args, stats):
                raise QuotaMet


def main():
    args = parse_args()
    keys_dir = Path(args.keys_dir) if args.keys_dir else \
        Path("./keys_desilo") / ("compact" if args.compact else "default")
    params = json.loads((keys_dir / "params.json").read_text())
    if bool(params.get("compact")) != args.compact:
        raise SystemExit(f"{keys_dir} built with compact={params.get('compact')}, "
                         f"target running compact={args.compact}.")

    # Transport first, then the slow init. Spawning workers and reading a key
    # set per worker takes minutes; the host opens its transport before doing
    # any of its own setup, so anything before this line is time the host
    # spends unable to reach us.
    link = None
    if not args.connect_after_init:
        link = transport_open(args)

    started = time.perf_counter()
    pool = Pool(args, keys_dir)

    if link is None:
        link = transport_open(args)

    if args.expect_bootstraps:
        print(f"Will exit automatically after {args.expect_bootstraps} bootstraps.")

    stats = {"received": 0, "served": 0}
    try:
        (serve_tcp if args.transport == "tcp" else serve_rdma)(
            args, link, pool, stats)
    except QuotaMet:
        print(f"\nServed {args.expect_bootstraps} bootstraps as requested; exiting.")
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