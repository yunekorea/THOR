"""
thor/he_ndp.py -- host-side bootstrap offloading, ported from src/thor/ckks_ndp.py.

WHAT CHANGED FROM THE LIBERATE VERSION
--------------------------------------
`CkksNDPEngine` subclassed `CkksEngine` and overrode `bootstrap()`. The wire
format was hand-rolled: it walked `DataStruct.data`, wrote a JSON header plus a
JSON tensor-manifest (shape/dtype/nbytes per tensor), then memcpy'd each GPU
tensor into the RDMA MR, and the far side rebuilt the DataStruct field by field.
About 160 lines that had to track Liberate's internal layout.

desilofhe serializes ciphertexts itself:

    buf = engine.serialize_ciphertext(ct)     -> bytearray
    ct  = engine.deserialize_ciphertext(buf)  -> Ciphertext

and `ct.serialized_nbytes` gives the exact byte count up front, which is what you
need to size an MR before allocating it. Verified on desilofhe 1.16: the buffer
length equals `serialized_nbytes`, and a round trip preserves both `level` and
plaintext to ~1e-6. So `ct_serialization` / `ct_deserialization` collapse to two
library calls and the payload becomes an opaque byte string -- which also means
the host and target no longer have to agree on tensor layout, only on the
library build.

The override point moved too. In the DESILO rewrite bootstrapping is one method
on `HE`:

    def bootstrap(self, ciphertext):
        return self.engine.bootstrap(ciphertext, relin, conj, bootstrap_key)

so `HENDP` subclasses `HE` rather than the engine, and `he.py` needs no changes.
Everything in `forward.py` that calls `he.bootstrap(...)` -- including the
internal calls from `he_inv`, `he_invsqrt`, `stage_07_softmax` and `stage_13`
-- routes to the target automatically.

TRANSPORTS
----------
`RdmaNvmeTransport` reproduces the original path: allocate an MR, hand its
rkey/addr/length to the NDP device through an NVMe passthrough command
(opcode 0xdb), let the target RDMA-read it, then wait for the result to arrive
by RDMA send. `TcpTransport` is a plain sockets fallback with the same request
/response shape, for validating correctness before the NDP hardware is in the
loop. Both move the same bytes.

UNVERIFIED HERE: the RDMA and NVMe paths need your hardware. They are a
structural port of code you already had working, not something I could execute.
"""

import socket
import struct
import time
from pathlib import Path

from desilofhe import Ciphertext

from .he import HE

# Payload framing. The response MR is a fixed size, so the header carries the
# true length -- same approach as the original _MAGIC preamble.
_MAGIC = b"DSLFHE01"
_HEADER = struct.Struct("<8sQ")          # magic, payload length
MAX_CT_BYTES = 64 * 1024 * 1024          # a level-14 ciphertext is ~10 MiB


def frame(payload: bytes) -> bytes:
    return _HEADER.pack(_MAGIC, len(payload)) + bytes(payload)


def unframe(buf: memoryview | bytes) -> bytes:
    magic, length = _HEADER.unpack_from(buf, 0)
    if magic != _MAGIC:
        raise ValueError(f"bad magic {magic!r}: corrupt or version mismatch")
    start = _HEADER.size
    return bytes(buf[start:start + length])


# ======================================================================
# Transports
# ======================================================================
class TcpTransport:
    """Length-prefixed request/response over TCP. Bring-up and correctness runs.

    Not the NDP datapath -- it exists so you can prove the split is numerically
    correct before RDMA and the NVMe passthrough are involved.
    """

    def __init__(self, host, port=9998):
        self.addr = (host, port)
        self.sock = socket.create_connection(self.addr)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def request(self, payload: bytes) -> bytes:
        self.sock.sendall(frame(payload))
        head = self._recv_exact(_HEADER.size)
        magic, length = _HEADER.unpack(head)
        if magic != _MAGIC:
            raise ValueError(f"bad magic {magic!r} in response")
        return self._recv_exact(length)

    def _recv_exact(self, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            b = self.sock.recv(min(1 << 20, n - got))
            if not b:
                raise ConnectionError(f"connection closed after {got}/{n} bytes")
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)

    def close(self):
        self.sock.close()


class RdmaNvmeTransport:
    """The NDP datapath: RDMA buffers, NVMe passthrough doorbell.

    Mirrors CkksNDPEngine.ct_serialization / nvme_passthru / receive_bs_result.
    The only real change is that the MR now holds an opaque serialized
    ciphertext instead of a JSON preamble plus raw tensors.
    """

    def __init__(self, host_ip="192.168.100.2", target_ip="192.168.100.1",
                 service="9999", nvme_dev="nvme1n1", ib_dev="enp216s0np0",
                 recv_bytes=MAX_CT_BYTES):
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

        cap = QPCap(max_send_wr=16, max_recv_wr=16, max_send_sge=8)
        qp_init_attr = QPInitAttr(cap=cap)
        sai = AddrInfo(src=host_ip, dst=target_ip, dst_service=service,
                       port_space=rdma_port_space.RDMA_PS_TCP)
        self.cmid = CMID(creator=sai, qp_init_attr=qp_init_attr)
        self.cmid.connect()

    def request(self, payload: bytes) -> bytes:
        ctypes = self._ctypes
        blob = frame(payload)

        # reg_read: the target RDMA-reads this buffer, so it needs remote-read rights.
        mr = self.cmid.reg_read(len(blob))
        (ctypes.c_uint8 * len(blob)).from_address(mr.buf)[:] = blob

        self._ring_doorbell(mr)

        rmr = self.cmid.reg_msgs(self.recv_bytes)
        self.cmid.post_recv(rmr)
        wc = self.cmid.get_recv_comp()
        if wc is None:
            raise RuntimeError("no recv completion for bootstrap result")

        view = bytes((ctypes.c_uint8 * self.recv_bytes).from_address(rmr.buf))
        result = unframe(view)
        mr.close()
        rmr.close()
        return result

    def _ring_doorbell(self, mr):
        """NVMe passthrough carrying rkey/addr/length/devname to the NDP device."""
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
        cmd.timeout_ms = 60000
        cmd.result = 0

        bufferptr = ctypes.c_void_p()
        if self.libc.posix_memalign(ctypes.byref(bufferptr),
                                    self.libc.getpagesize(), 4096) != 0:
            raise MemoryError("posix_memalign failed for passthru buffer")
        ctypes.memset(bufferptr, 0, 4096)
        cmd.data = bufferptr.value

        packed = struct.pack(f"<QQII{dev_name_len}s", mr.rkey, mr.buf,
                             mr.length, dev_name_len, self.ib_dev)
        ctypes.memmove(bufferptr.value, packed, len(packed))
        return nvme.ndp_passthru(self.fd, cmd)

    def close(self):
        try:
            self.cmid.close()
        except Exception:
            pass


# ======================================================================
# Host-side HE with offloaded bootstrapping
# ======================================================================
class HENDP(HE):
    """HE whose bootstrap() runs on the NDP target instead of locally.

    The host loads a general RotationKey instead of the bootstrap key. THOR uses
    the bootstrap key as a ROTATION key for the ~20 forward-pass deltas that fall
    inside bootstrap_deltas (see HE.rotate) -- those rotations are ordinary layer
    work, not bootstrapping, and a RotationKey serves them identically at a
    fraction of the size. This is the direct analogue of the Liberate split, where
    the host loaded the Galois key `gk` and left `rotk_dict` on the target.

    Set skip_bootstrap_key=False to load the bootstrap key locally as well, for
    A/B runs against local bootstrapping.
    """

    def __init__(self, device, compact, bootstrap_key_size, timer,
                 keys_dir=None, mode="gpu", transport=None,
                 skip_bootstrap_key=True, use_rotation_key=True, verbose=True):
        self._skip_bootstrap_key = skip_bootstrap_key
        super().__init__(device, compact, bootstrap_key_size, timer,
                         keys_dir=keys_dir, mode=mode,
                         use_rotation_key=use_rotation_key)
        self.transport = transport
        self.verbose = verbose
        self.bootstrap_calls = 0
        self.bootstrap_seconds = 0.0
        self.serialize_seconds = 0.0
        self.transport_seconds = 0.0
        self.bytes_sent = 0
        self.bytes_received = 0

    def bootstrap(self, ciphertext: Ciphertext) -> Ciphertext:
        if self.transport is None:
            raise RuntimeError(
                "HENDP has no transport; construct it with transport=... or use HE."
            )
        self.bootstrap_calls += 1
        n = self.bootstrap_calls
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        payload = self.engine.serialize_ciphertext(ciphertext)
        self.serialize_seconds += time.perf_counter() - t0
        self.bytes_sent += len(payload)

        t0 = time.perf_counter()
        reply = self.transport.request(payload)
        self.transport_seconds += time.perf_counter() - t0
        self.bytes_received += len(reply)

        t0 = time.perf_counter()
        result = self.engine.deserialize_ciphertext(reply)
        self.serialize_seconds += time.perf_counter() - t0

        dt = time.perf_counter() - t_all
        self.bootstrap_seconds += dt
        if self.verbose:
            print(f"  [bs #{n:04d}] offloaded {len(payload)/1024**2:.1f}MiB -> "
                  f"{len(reply)/1024**2:.1f}MiB in {dt:.2f}s "
                  f"(level {ciphertext.level} -> {result.level})")
        return result

    def report(self):
        return dict(
            bootstrap_calls=self.bootstrap_calls,
            bootstrap_seconds=round(self.bootstrap_seconds, 3),
            serialize_seconds=round(self.serialize_seconds, 3),
            transport_seconds=round(self.transport_seconds, 3),
            mib_sent=round(self.bytes_sent / 1024**2, 1),
            mib_received=round(self.bytes_received / 1024**2, 1),
        )


def make_transport(kind, args):
    if kind == "tcp":
        return TcpTransport(args.target_ip, args.port)
    if kind == "rdma":
        return RdmaNvmeTransport(host_ip=args.host_ip, target_ip=args.target_ip,
                                 nvme_dev=args.nvme_dev, ib_dev=args.ib_dev)
    raise ValueError(f"unknown transport {kind!r}")