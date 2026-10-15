"""
thor/ndp_parallel.py -- shared pieces for the parallel host/target pair.

Why this is separate from he_ndp.py: concurrency changes the wire format. With
several inferences in flight there is nothing in a reply that says which request
it answers, so every frame carries a request id. That makes the parallel host and
parallel target a matched pair -- they cannot talk to the single-stream versions,
and the single-stream versions keep working untouched.

Three problems concurrency creates, and what solves each:

1. desilofhe's Engine is NOT safe for concurrent calls on one instance. Six
   threads sharing one engine segfaulted here; the identical workload run
   serially was fine. On the host there is one GPU, so serializing engine calls
   costs nothing -- LockedEngine wraps every call in one lock (the "engine
   lock"). The overlap we actually want is compute against network wait, and
   bootstrap() releases the lock before going to the wire.

   How long the lock is held is a choice. By default it is taken per engine
   call, so streams interleave call by call. With --gpu-token (GpuLock) a
   stream holds it for its whole GPU phase and hands it over only during the
   bootstrap wire wait and while its next weights are read from disk.

2. thor.timer.Timer is stateful (it nests pause/resume), so concurrent streams
   would corrupt each other's readings. StageTimer keeps all of its state per
   thread and records every stage of every layer for every stream.

3. Replies arrive out of order. MuxTransport keeps one slot per in-flight
   request and a dedicated reader thread that wakes the right waiter.
"""

import itertools
import os
import sys
import struct
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

from .he import HE
from .stage_timer import StageTimer

# magic, request id, payload length
PHEADER = struct.Struct("<8sQQ")
PMAGIC = b"DSLPAR01"
MAX_CT_BYTES = 64 * 1024 * 1024


def pframe(req_id: int, payload: bytes) -> bytes:
    return PHEADER.pack(PMAGIC, req_id, len(payload)) + bytes(payload)


def punframe(buf) -> tuple[int, bytes]:
    magic, req_id, length = PHEADER.unpack_from(buf, 0)
    if magic != PMAGIC:
        raise ValueError(f"bad magic {magic!r}: parallel host needs the parallel target")
    start = PHEADER.size
    return req_id, bytes(buf[start:start + length])


# ======================================================================
# Timer stand-in
# ======================================================================
class NullTimer:
    """Same surface as thor.timer.Timer, records nothing.

    forward_layer does `timer = he.timer` and opens stage/layer contexts, so the
    object has to exist -- but a single shared Timer across N streams would
    interleave its pause/resume nesting and report nonsense.
    """

    def elapsed(self):
        return "0.000s"

    def true_elapsed(self):
        return "0.000s"

    def reset(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def print_legend(self):
        pass

    @contextmanager
    def paused(self):
        yield

    @contextmanager
    def setup(self):
        yield

    @contextmanager
    def stage(self, stage_index, stage_name):
        yield

    @contextmanager
    def layer(self, layer_index):
        yield


# ======================================================================
# Engine serialization
# ======================================================================
class LockedEngine:
    """Serializes every engine call behind one RLock.

    Coarse on purpose. There is one GPU, so engine work was always going to
    serialize; the lock just makes it safe. Non-callable attributes
    (slot_count, max_level, build_hash) pass straight through.
    """

    def __init__(self, engine, lock, on_wait=None):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "_on_wait", on_wait)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_engine"), name)
        if not callable(attr):
            return attr
        lock = object.__getattribute__(self, "_lock")
        on_wait = object.__getattribute__(self, "_on_wait")

        def locked(*args, **kwargs):
            # Try first without blocking: almost always free (or already ours,
            # with --gpu-token), and then there is nothing to time. Only a real
            # wait -- another stream inside the engine -- is measured and
            # reported, which is what makes the default mode's contention
            # comparable with --gpu-token's handoff waits.
            if not lock.acquire(blocking=False):
                t0 = time.perf_counter()
                lock.acquire()
                if on_wait is not None:
                    on_wait(time.perf_counter() - t0)
            try:
                return attr(*args, **kwargs)
            finally:
                lock.release()

        return locked

    @property
    def raw(self):
        return object.__getattribute__(self, "_engine")


# ======================================================================
# GPU token (--gpu-token)
# ======================================================================
class GpuLock:
    """The engine lock, held per GPU phase instead of per engine call.

    It is still THE engine lock -- LockedEngine and rotate() take it exactly as
    before -- but with --gpu-token each stream takes it once when it starts and
    keeps it. Every engine call then re-enters a lock the stream already owns,
    so no other stream can get in between two of its calls. The stream gives it
    up only inside yielded(), which HENDPParallel uses at exactly two places:

      1. the bootstrap wire wait (request sent, reply not yet back), and
      2. reading a stage's weight files into the page cache.

    Neither window contains an engine call, so the engine is never entered by
    two threads at once.

    Why not threading.RLock: yielded() has to drop the lock completely however
    deeply the stream holds it (it may be inside rotate(), inside LockedEngine)
    and restore that exact depth afterwards. RLock has no public way to do
    that, so this tracks owner and depth itself.

    Waiters are served in arrival order. Python's own locks are not fair: the
    thread that just released often takes the lock straight back. With a
    handoff at only two points per GPU phase, that would let one stream run
    several phases in a row while another's bootstrap reply sits waiting.
    """

    def __init__(self):
        self._cv = threading.Condition(threading.Lock())
        self._owner = None
        self._depth = 0
        self._queue = deque()

    def _take(self, me):
        # caller holds self._cv
        ticket = object()
        self._queue.append(ticket)
        while self._owner is not None or self._queue[0] is not ticket:
            self._cv.wait()
        self._queue.popleft()
        self._owner = me
        if self._queue:
            # the next waiter must re-check: the queue head has changed
            self._cv.notify_all()

    def acquire(self, blocking=True):
        me = threading.get_ident()
        with self._cv:
            if self._owner == me:
                self._depth += 1
                return True
            if not blocking:
                if self._owner is None and not self._queue:
                    self._owner, self._depth = me, 1
                    return True
                return False
            self._take(me)
            self._depth = 1
            return True

    def release(self):
        with self._cv:
            if self._owner != threading.get_ident():
                raise RuntimeError("GpuLock released by a thread that does not hold it")
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._cv.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def held_by_me(self):
        with self._cv:
            return self._owner == threading.get_ident()

    @contextmanager
    def yielded(self):
        """Give the lock away for the duration of the block, then take it back.

        Yields a dict; after the block, info["wait"] is how long the stream
        waited to get the lock back, i.e. how long the other stream kept it
        after this stream's I/O or bootstrap was already finished. A no-op
        (wait 0) if the calling thread does not hold the lock.
        """
        me = threading.get_ident()
        info = {"wait": 0.0, "yielded": False}
        with self._cv:
            depth = self._depth if self._owner == me else 0
            if depth:
                self._owner, self._depth = None, 0
                self._cv.notify_all()
        try:
            yield info
        finally:
            if depth:
                t0 = time.perf_counter()
                with self._cv:
                    self._take(me)
                    self._depth = depth
                info["wait"] = time.perf_counter() - t0
                info["yielded"] = True


def prefetch_dir(path, threads=1, chunk=1 << 20):
    """Read every file in `path` once so the next reads hit the page cache.

    Plain file reads, no engine involved -- which is the whole point: this is
    the slow half of weight loading (the disk / NVMe-oF read), split off so it
    can run while another stream holds the GPU. Returns bytes read. A missing
    directory is not an error here; the engine read that follows will report it
    properly.
    """
    try:
        names = [e.path for e in os.scandir(path) if e.is_file()]
    except FileNotFoundError:
        return 0

    def read_all(paths):
        buf = bytearray(chunk)
        view = memoryview(buf)
        total = 0
        for p in paths:
            with open(p, "rb", buffering=0) as fh:
                while True:
                    n = fh.readinto(view)
                    if not n:
                        break
                    total += n
        return total

    if threads <= 1 or len(names) < 2:
        return read_all(names)
    parts = [names[i::threads] for i in range(threads)]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return sum(pool.map(read_all, parts))


# ======================================================================
# Multiplexed transports
# ======================================================================
class _MuxBase:
    """Request/response multiplexing over a single connection."""

    def __init__(self, timeout=1800.0):
        self._ids = itertools.count(1)
        self._slots = {}
        self._slots_lock = threading.Lock()
        self._closed = False
        self._timeout = timeout
        self.error = None

    def _new_slot(self):
        req_id = next(self._ids)
        event, box = threading.Event(), [None]
        with self._slots_lock:
            self._slots[req_id] = (event, box)
        return req_id, event, box

    def _deliver(self, req_id, payload):
        with self._slots_lock:
            slot = self._slots.pop(req_id, None)
        if slot is None:
            return
        event, box = slot
        box[0] = payload
        event.set()

    def _fail_all(self, exc):
        self.error = exc
        with self._slots_lock:
            slots = list(self._slots.values())
            self._slots.clear()
        for event, box in slots:
            box[0] = None
            event.set()

    def _wait(self, req_id, event, box):
        if not event.wait(self._timeout):
            with self._slots_lock:
                self._slots.pop(req_id, None)
            raise TimeoutError(f"no reply for request {req_id} within {self._timeout}s")
        if box[0] is None:
            raise ConnectionError(f"transport failed while request {req_id} was in flight: "
                                  f"{self.error}")
        return box[0]


class RdmaMuxTransport(_MuxBase):
    """NDP datapath with request ids. UNTESTED -- needs your hardware.

    Same shape as he_ndp.RdmaNvmeTransport: register an MR, ring the NVMe
    doorbell, let the target RDMA-read it. The differences are that several MRs
    are live at once (one per in-flight stream) and a reader thread drains
    completions and wakes waiters by id, instead of each caller blocking on
    get_recv_comp itself.

    The MR must stay registered until its reply arrives -- the target reads it
    asynchronously -- so it is closed after the wait, not before.
    """

    # The Liberate host's fixed receive buffer, which worked on this hardware.
    LIBERATE_RECV_BYTES = 52428800

    def __init__(self, host_ip="192.168.100.2", target_ip="192.168.100.1",
                 service="9999", nvme_dev="nvme1n1", ib_dev="enp216s0np0",
                 recv_slots=8, recv_bytes=MAX_CT_BYTES, timeout=1800.0,
                 recv_mode="pool"):
        super().__init__(timeout)
        # recv_mode="per-request" reproduces the working single-stream host
        # exactly: one fixed 50 MiB receive, posted AFTER the doorbell, reaped by
        # the calling thread. Only meaningful with one stream -- it bypasses the
        # multiplexer -- and it exists to split host from target when debugging.
        self.recv_mode = recv_mode
        import ctypes
        from ctypes.util import find_library

        from libnvme import nvme
        from pyverbs.cmid import CMID, AddrInfo
        from pyverbs.librdmacm_enums import rdma_port_space
        from pyverbs.qp import QPCap, QPInitAttr

        self._ctypes = ctypes
        self._nvme = nvme
        self.libc = ctypes.CDLL(find_library("c"))
        self.fd = nvme.nvme_open(nvme_dev)
        self.ib_dev = ib_dev.encode("utf-8")
        self.recv_bytes = recv_bytes

        cap = QPCap(max_send_wr=64, max_recv_wr=64, max_send_sge=8)
        sai = AddrInfo(src=host_ip, dst=target_ip, dst_service=service,
                       port_space=rdma_port_space.RDMA_PS_TCP)
        self.cmid = CMID(creator=sai, qp_init_attr=QPInitAttr(cap=cap))
        self.cmid.connect()

        self._cmid_lock = threading.Lock()
        self._recv_mrs = []

        if self.recv_mode == "per-request":
            print("  RDMA recv mode: per-request (one 50 MiB buffer after each "
                  "doorbell, as the working host does)")
            return

        # Pre-post a pool of receive buffers so replies always have somewhere to
        # land; one per concurrent stream, plus slack.
        for _ in range(recv_slots):
            mr = self.cmid.reg_msgs(recv_bytes)
            self.cmid.post_recv(mr)
            self._recv_mrs.append(mr)
        print(f"  RDMA recv mode: pool ({recv_slots} x "
              f"{recv_bytes / 1024**2:.0f} MiB pre-posted)")

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        ctypes = self._ctypes
        try:
            idx = 0
            while not self._closed:
                wc = self.cmid.get_recv_comp()
                if wc is None:
                    raise RuntimeError("no recv completion")
                mr = self._recv_mrs[idx % len(self._recv_mrs)]
                view = bytes((ctypes.c_uint8 * self.recv_bytes).from_address(mr.buf))
                req_id, payload = punframe(view)
                with self._cmid_lock:
                    self.cmid.post_recv(mr)
                idx += 1
                self._deliver(req_id, payload)
        except Exception as exc:
            if not self._closed:
                self._fail_all(exc)

    def request(self, payload: bytes) -> bytes:
        if self.recv_mode == "per-request":
            return self._request_per_request(payload)
        ctypes = self._ctypes
        req_id, event, box = self._new_slot()
        blob = pframe(req_id, payload)

        with self._cmid_lock:
            mr = self.cmid.reg_read(len(blob))
            (ctypes.c_uint8 * len(blob)).from_address(mr.buf)[:] = blob
            self._ring_doorbell(mr)
        try:
            return self._wait(req_id, event, box)
        finally:
            with self._cmid_lock:
                mr.close()

    def _request_per_request(self, payload: bytes) -> bytes:
        """Line-for-line the working host: reg_read, doorbell, then recv."""
        ctypes = self._ctypes
        req_id = next(self._ids)
        blob = pframe(req_id, payload)

        mr = self.cmid.reg_read(len(blob))
        (ctypes.c_uint8 * len(blob)).from_address(mr.buf)[:] = blob
        self._ring_doorbell(mr)

        rmr = self.cmid.reg_msgs(self.LIBERATE_RECV_BYTES)
        self.cmid.post_recv(rmr)
        wc = self.cmid.get_recv_comp()
        if wc is None:
            raise RuntimeError("no recv completion")
        view = bytes((ctypes.c_uint8 * self.LIBERATE_RECV_BYTES).from_address(rmr.buf))
        got_id, reply = punframe(view)
        mr.close()
        rmr.close()
        if got_id != req_id:
            raise RuntimeError(f"reply for {got_id}, expected {req_id}")
        return reply

    def _ring_doorbell(self, mr):
        ctypes, nvme = self._ctypes, self._nvme
        dev_name_len = len(self.ib_dev)

        cmd = nvme.ndp_passthru_cmd()
        cmd.opcode = 0xDB
        cmd.flags = cmd.rsvd = 0
        cmd.nsid = 1
        for f in ("cdw2", "cdw3", "cdw10", "cdw11", "cdw12", "cdw13", "cdw14", "cdw15"):
            setattr(cmd, f, 0)
        cmd.data_len = 4096
        cmd.metadata_len = cmd.metadata = 0
        cmd.timeout_ms = 600000
        cmd.result = 0

        bufferptr = ctypes.c_void_p()
        if self.libc.posix_memalign(ctypes.byref(bufferptr),
                                    self.libc.getpagesize(), 4096) != 0:
            raise MemoryError("posix_memalign failed")
        ctypes.memset(bufferptr, 0, 4096)
        cmd.data = bufferptr.value

        packed = struct.pack(f"<QQII{dev_name_len}s", mr.rkey, mr.buf,
                             mr.length, dev_name_len, self.ib_dev)
        ctypes.memmove(bufferptr.value, packed, len(packed))
        return nvme.ndp_passthru(self.fd, cmd)

    def close(self):
        self._closed = True
        try:
            self.cmid.close()
        except Exception:
            pass


def make_parallel_transport(kind, args):
    # RDMA is the only transport: there is no practical way to move ciphertexts
    # over TCP in this setup, so the TCP fallback was removed.
    if kind == "rdma":
        recv_mode = getattr(args, "recv_mode", "pool")
        if recv_mode == "per-request" and args.streams != 1:
            raise SystemExit("--recv-mode per-request only works with --streams 1: "
                             "it bypasses the multiplexer, so replies cannot be "
                             "matched to streams.")
        return RdmaMuxTransport(host_ip=args.host_ip, target_ip=args.target_ip,
                                nvme_dev=args.nvme_dev, ib_dev=args.ib_dev,
                                recv_slots=max(8, args.streams * 2),
                                recv_mode=recv_mode)
    raise ValueError(f"unknown transport {kind!r}")


# ======================================================================
# Host-side HE, shared by all streams
# ======================================================================
class HENDPParallel(HE):
    """One HE, one set of keys, N inference threads.

    Keys are shared because they are the expensive part -- duplicating them per
    stream would defeat the VRAM saving the NDP split buys. Safety comes from
    LockedEngine plus an override of rotate(), which mutates
    fixed_rotation_keys / rotate_levels at runtime and so needs the same lock
    across the whole method (hence RLock: rotate holds it, then calls into the
    already-locked engine).
    """

    def __init__(self, device, compact, bootstrap_key_size, keys_dir=None,
                 mode="gpu", transport=None, use_rotation_key=True,
                 skip_bootstrap_key=True, verbose=True, timer=None,
                 gpu_token=False, prefetch_threads=1):
        self._skip_bootstrap_key = skip_bootstrap_key
        # Default: the original fine-grained scheme, an RLock taken per engine
        # call. --gpu-token: the same lock, but held for a whole GPU phase and
        # handed over only at bootstrap and weight-loading points (GpuLock).
        self.gpu_token = gpu_token
        self.prefetch_threads = max(1, prefetch_threads)
        self.lock = GpuLock() if gpu_token else threading.RLock()
        timer = timer if timer is not None else StageTimer()
        super().__init__(device, compact, bootstrap_key_size, timer,
                         keys_dir=keys_dir, mode=mode,
                         use_rotation_key=use_rotation_key)
        # wrap only after __init__ has finished building keys
        self.engine = LockedEngine(self.engine, self.lock,
                                   on_wait=self._engine_lock_waited)
        if gpu_token:
            self._install_weight_prefetch()
        self.transport = transport
        self.verbose = verbose
        self._tls = threading.local()

        self._stats_lock = threading.Lock()
        # print() emits the text and the newline as separate writes, so with
        # several streams logging at once the lines merge into each other.
        self._print_lock = threading.Lock()
        self.bootstrap_calls = 0
        self.bootstrap_seconds = 0.0
        self.serialize_seconds = 0.0
        self.transport_seconds = 0.0
        self.bytes_sent = 0
        self.bytes_received = 0
        # per-stream, keyed by thread name -- with several inferences in flight a
        # bare total hides whether one stream is being starved of the GPU lock
        self.per_stream = {}

    # ------------------------------------------------------------ the lock
    def stream_turn(self):
        """Wrap a stream's whole inference in this.

        With --gpu-token the stream takes the lock here and keeps it until the
        inference ends (released on error too); without, it is a no-op and
        locking stays per engine call.
        """
        return self.lock if self.gpu_token else nullcontext()

    def _stream_rec(self):
        # caller holds self._stats_lock
        return self.per_stream.setdefault(
            threading.current_thread().name,
            {"calls": 0, "seconds": 0.0, "wire_seconds": 0.0,
             "engine_lock_wait": 0.0, "engine_lock_waits": 0,
             "token_wait_bootstrap": 0.0, "token_wait_weights": 0.0,
             "token_wait_max": 0.0, "handoffs": 0,
             "prefetch_seconds": 0.0, "prefetch_bytes": 0})

    def _engine_lock_waited(self, seconds):
        # LockedEngine calls this only when an engine call actually had to wait
        # for another stream. With --gpu-token that cannot happen during a
        # stream's own phase (it already holds the lock), so this measures the
        # default mode's call-by-call contention.
        with self._stats_lock:
            rec = self._stream_rec()
            rec["engine_lock_wait"] += seconds
            rec["engine_lock_waits"] += 1
        self.timer.note_wait(seconds)

    def _note_handoff(self, kind, wait):
        with self._stats_lock:
            rec = self._stream_rec()
            rec["handoffs"] += 1
            rec[f"token_wait_{kind}"] += wait
            rec["token_wait_max"] = max(rec["token_wait_max"], wait)

    def _install_weight_prefetch(self):
        """Second handoff point: weight loading.

        Every per-layer weight load in he.py is a block of
        engine.read_light_plaintext() calls on files in ONE directory (one
        stage of one layer). read_light_plaintext is an engine call, so the lock
        cannot be given up around it. Instead, on the first read from a new
        directory, the stream gives the lock up, reads the whole directory into
        the page cache with plain file I/O, takes the lock back, and lets the
        engine reads that follow come from memory. One handoff per weight
        block, and he.py is untouched.
        """
        locked_read = self.engine.read_light_plaintext   # the normal locked call

        def read_light_plaintext(path):
            if self.lock.held_by_me():
                d = os.path.dirname(str(path))
                seen = getattr(self._tls, "prefetched", None)
                if seen is None:
                    seen = self._tls.prefetched = set()
                if d not in seen:
                    seen.add(d)
                    with self.lock.yielded() as info:
                        t0 = time.perf_counter()
                        nbytes = prefetch_dir(d, self.prefetch_threads)
                        io = time.perf_counter() - t0
                    self._note_handoff("weights", info["wait"])
                    with self._stats_lock:
                        rec = self._stream_rec()
                        rec["prefetch_seconds"] += io
                        rec["prefetch_bytes"] += nbytes
                    self.timer.note_wait(info["wait"])
            return locked_read(path)

        # an instance attribute wins over LockedEngine.__getattr__
        object.__setattr__(self.engine, "read_light_plaintext", read_light_plaintext)

    def rotate(self, ciphertext, delta):
        with self.lock:
            return super().rotate(ciphertext, delta)

    def bootstrap(self, ciphertext):
        import time as _time
        if self.transport is None:
            raise RuntimeError("HENDPParallel needs a transport")

        stream = threading.current_thread().name
        level_in = getattr(ciphertext, "level", None)
        t_all = _time.perf_counter()

        # serialize under the engine lock (LockedEngine takes it for us)
        t0 = _time.perf_counter()
        payload = self.engine.serialize_ciphertext(ciphertext)
        ser = _time.perf_counter() - t0

        # The wire wait happens WITHOUT the lock -- this is where another stream
        # gets to use the GPU. Default mode: nothing to do, the lock was already
        # dropped when serialize_ciphertext returned. --gpu-token: the stream
        # owns the lock for its whole phase, so give it away explicitly here and
        # take it back once the reply is in. In both modes the reply itself is
        # reaped by the transport's reader thread, which never needs the lock.
        handoff = self.lock.yielded() if self.gpu_token else nullcontext({"wait": 0.0})
        with handoff as info:
            t0 = _time.perf_counter()
            reply = self.transport.request(payload)
            wire = _time.perf_counter() - t0
        token_wait = info["wait"]

        t0 = _time.perf_counter()
        result = self.engine.deserialize_ciphertext(reply)
        ser += _time.perf_counter() - t0

        dt = _time.perf_counter() - t_all
        with self._stats_lock:
            self.bootstrap_calls += 1
            n = self.bootstrap_calls
            self.bootstrap_seconds += dt
            self.serialize_seconds += ser
            self.transport_seconds += wire
            self.bytes_sent += len(payload)
            self.bytes_received += len(reply)
            rec = self._stream_rec()
            rec["calls"] += 1
            rec["seconds"] += dt
            rec["wire_seconds"] += wire
        if self.gpu_token:
            self._note_handoff("bootstrap", token_wait)
        self.timer.note_bootstrap(dt, wait=token_wait)

        if self.verbose:
            tw = f", token wait {token_wait:.2f}s" if self.gpu_token else ""
            line = (f"  [{stream}] [bs #{n:04d}] offloaded "
                    f"{len(payload) / 1024**2:.1f} MiB -> "
                    f"{len(reply) / 1024**2:.1f} MiB in {dt:.2f}s "
                    f"(wire {wire:.2f}s{tw}; "
                    f"level {level_in} -> {getattr(result, 'level', None)})\n")
            with self._print_lock:
                sys.stdout.write(line)
                sys.stdout.flush()
        return result

    def report(self):
        with self._stats_lock:
            calls = self.bootstrap_calls
            return dict(
                bootstrap_calls=calls,
                bootstrap_seconds=round(self.bootstrap_seconds, 3),
                bootstrap_mean_seconds=(round(self.bootstrap_seconds / calls, 3)
                                        if calls else 0.0),
                serialize_seconds=round(self.serialize_seconds, 3),
                transport_seconds=round(self.transport_seconds, 3),
                mib_sent=round(self.bytes_sent / 1024**2, 1),
                mib_received=round(self.bytes_received / 1024**2, 1),
                bootstrap_per_stream={
                    k: {"calls": v["calls"],
                        "seconds": round(v["seconds"], 3),
                        "mean_seconds": (round(v["seconds"] / v["calls"], 3)
                                         if v["calls"] else 0.0)}
                    for k, v in sorted(self.per_stream.items())
                },
                lock_mode="gpu-token" if self.gpu_token else "per-call",
                # Time each stream lost waiting for the other(s). Default mode:
                # engine calls blocked by another stream's call. --gpu-token:
                # time to get the token back after a bootstrap reply arrived or
                # a weight prefetch finished.
                waiting_per_stream={
                    k: {"engine_lock_wait": round(v["engine_lock_wait"], 3),
                        "engine_lock_waits": v["engine_lock_waits"],
                        "token_wait_bootstrap": round(v["token_wait_bootstrap"], 3),
                        "token_wait_weights": round(v["token_wait_weights"], 3),
                        "token_wait_max": round(v["token_wait_max"], 3),
                        "handoffs": v["handoffs"],
                        "prefetch_seconds": round(v["prefetch_seconds"], 3),
                        "prefetch_gib": round(v["prefetch_bytes"] / 1024**3, 2)}
                    for k, v in sorted(self.per_stream.items())
                },
            )
