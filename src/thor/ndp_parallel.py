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
   costs nothing -- LockedEngine wraps every call in one RLock. The overlap we
   actually want is compute against network wait, and bootstrap() releases the
   lock before going to the wire.

2. Timer is stateful (it nests pause/resume), so concurrent streams would
   corrupt each other's readings. NullTimer satisfies the same interface and
   records nothing; the parallel host times whole inferences instead.

3. Replies arrive out of order. MuxTransport keeps one slot per in-flight
   request and a dedicated reader thread that wakes the right waiter.
"""

import itertools
import socket
import sys
import struct
import threading
from contextlib import contextmanager

from .he import HE

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


def recv_exact(sock, n):
    chunks, got = [], 0
    while got < n:
        b = sock.recv(min(1 << 20, n - got))
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


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

    def __init__(self, engine, lock):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_lock", lock)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_engine"), name)
        if not callable(attr):
            return attr
        lock = object.__getattribute__(self, "_lock")

        def locked(*args, **kwargs):
            with lock:
                return attr(*args, **kwargs)

        return locked

    @property
    def raw(self):
        return object.__getattribute__(self, "_engine")


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


class TcpMuxTransport(_MuxBase):
    """Length-prefixed frames with request ids over one TCP connection.

    Use this for bring-up: it exercises the whole parallel path without RDMA.
    """

    def __init__(self, host, port=9998, timeout=1800.0):
        super().__init__(timeout)
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._send_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        try:
            while not self._closed:
                head = recv_exact(self.sock, PHEADER.size)
                if head is None:
                    raise ConnectionError("target closed the connection")
                magic, req_id, length = PHEADER.unpack(head)
                if magic != PMAGIC:
                    raise ValueError(f"bad magic {magic!r}")
                payload = recv_exact(self.sock, length)
                if payload is None:
                    raise ConnectionError("target closed mid-frame")
                self._deliver(req_id, payload)
        except Exception as exc:
            if not self._closed:
                self._fail_all(exc)

    def request(self, payload: bytes) -> bytes:
        req_id, event, box = self._new_slot()
        with self._send_lock:
            self.sock.sendall(pframe(req_id, payload))
        return self._wait(req_id, event, box)

    def close(self):
        self._closed = True
        try:
            self.sock.close()
        except Exception:
            pass


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
    if kind == "tcp":
        return TcpMuxTransport(args.target_ip, args.port)
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
                 skip_bootstrap_key=True, verbose=True):
        self._skip_bootstrap_key = skip_bootstrap_key
        self.lock = threading.RLock()
        super().__init__(device, compact, bootstrap_key_size, NullTimer(),
                         keys_dir=keys_dir, mode=mode,
                         use_rotation_key=use_rotation_key)
        # wrap only after __init__ has finished building keys
        self.engine = LockedEngine(self.engine, self.lock)
        self.transport = transport
        self.verbose = verbose

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

        # the wire wait happens WITHOUT the lock -- this is the whole point,
        # it is where another stream gets to use the GPU
        t0 = _time.perf_counter()
        reply = self.transport.request(payload)
        wire = _time.perf_counter() - t0

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
            rec = self.per_stream.setdefault(
                stream, {"calls": 0, "seconds": 0.0, "wire_seconds": 0.0})
            rec["calls"] += 1
            rec["seconds"] += dt
            rec["wire_seconds"] += wire

        if self.verbose:
            line = (f"  [{stream}] [bs #{n:04d}] offloaded "
                    f"{len(payload) / 1024**2:.1f} MiB -> "
                    f"{len(reply) / 1024**2:.1f} MiB in {dt:.2f}s "
                    f"(level {level_in} -> {getattr(result, 'level', None)})\n")
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
            )