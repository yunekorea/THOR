import torch
from liberate.fhe.data_struct import DataStruct

from thor import CkksEngine

from pyverbs.cmid import CMID, AddrInfo
from pyverbs.qp import QPInitAttr, QPCap, QPAttr, QP
from pyverbs.libibverbs_enums import ibv_qp_type
from pyverbs.librdmacm_enums import rdma_port_space, RAI_PASSIVE
from pyverbs.cq import CQ

import json
import math
import struct

from libnvme import nvme

import ctypes
from ctypes.util import find_library

_MAGIC = b"LBFHE001"
_ALIGN  = 64   # align tensor data to 64-byte boundary (cache-line friendly)
def _align_up(n: int, align: int) -> int:
    return int(math.ceil(n / align) * align)

class CkksNDPEngine(CkksEngine):
    def __init__(self, cmid:CMID, params, verbose=False):
        """
        CustomCkksEngine is a child class of CkksEngine that overrides the bootstrap method
        with a custom implementation.
        """
        super().__init__(params)
        self.libc = ctypes.CDLL(find_library('c'))
        self.fd = nvme.nvme_open("nvme1n1")
        self.cmid = cmid

    def ct_serialization(self, ct: DataStruct):
        # ── 1. Build header JSON ──────────────────────────────────────────────────
        header = {
            "include_special":  ct.include_special,
            "ntt_state":        ct.ntt_state,
            "montgomery_state": ct.montgomery_state,
            "origin":           ct.origin,
            "level_calc":       ct.level_calc,
            "level_available":  ct.level_available,
            "hash":             ct.hash,
            "version":          ct.version,
        }
        header_bytes = json.dumps(header).encode("utf-8")
        
        #print(f"[Serialize] DataStruct BEFORE Serialization(Host - before BS): "
        #    f"level={ct.level}, level_calc={ct.level_calc}, level_avil={ct.level_available}")

        # ── 2. Build tensor-meta JSON (nested list mirrors ct.data structure) ─────
        tensor_meta = []
        total_tensor_bytes = 0
        for i, poly_list in enumerate(ct.data):
            poly_meta = []
            for j, tensor in enumerate(poly_list):
                if not tensor.is_cuda:
                    raise ValueError(f"Tensor [{i}][{j}] must be on the GPU before serialization.")
                nbytes = tensor.nelement() * tensor.element_size()
                poly_meta.append({
                    "shape":  list(tensor.shape),
                    "dtype":  str(tensor.dtype).replace("torch.", ""),
                    "nbytes": nbytes,
                })
                total_tensor_bytes += nbytes
            tensor_meta.append(poly_meta)
        tensor_meta_bytes = json.dumps(tensor_meta).encode("utf-8")

        # ── 3. Calculate total MR size ────────────────────────────────────────────
        preamble_size = (
            len(_MAGIC)
            + 8                           # header_json_len
            + 8                           # tensor_meta_json_len
            + len(header_bytes)
            + len(tensor_meta_bytes)
        )
        tensor_data_offset = _align_up(preamble_size, _ALIGN)
        total_mr_size      = tensor_data_offset + total_tensor_bytes

        #print(f"[Step 1] MR layout: {preamble_size}B preamble + "
        #      f"{tensor_data_offset - preamble_size}B padding + "
        #      f"{total_tensor_bytes}B tensor payload = {total_mr_size}B total")

        # ── 4. Allocate RDMA MR ───────────────────────────────────────────────────
        mr         = self.cmid.reg_read(total_mr_size)
        mr_pointer = mr.buf
        #print(f"[Step 2] MR allocated at host address: {hex(mr_pointer)}")

        # ── 5. Write preamble into MR ─────────────────────────────────────────────
        preamble = (
            _MAGIC
            + struct.pack("<Q", len(header_bytes))
            + struct.pack("<Q", len(tensor_meta_bytes))
            + header_bytes
            + tensor_meta_bytes
        )
        preamble_arr = (ctypes.c_uint8 * len(preamble)).from_address(mr_pointer)
        preamble_arr[:] = preamble
        #print(f"[Step 3] Preamble written ({len(preamble)}B)")

        # ── 6. Copy tensors GPU → MR ──────────────────────────────────────────────
        current_offset = tensor_data_offset
        for poly_list, poly_meta in zip(ct.data, tensor_meta):
            for tensor, meta in zip(poly_list, poly_meta):
                nbytes     = meta["nbytes"]
                chunk_addr = mr_pointer + current_offset
                ctypes_arr = (ctypes.c_uint8 * nbytes).from_address(chunk_addr)
                host_view  = (torch.frombuffer(ctypes_arr, dtype=torch.uint8)
                                .view(tensor.dtype)
                                .view(tensor.shape))
                host_view.copy_(tensor, non_blocking=True)
                current_offset += nbytes

        torch.cuda.synchronize()
        #print(f"[Step 4] Tensors copied GPU → MR.")

        return mr, total_mr_size


    # ──────────────────────────────────────────────────────────────────────────────
    # Deserialization
    # ──────────────────────────────────────────────────────────────────────────────

    def ct_deserialization(self, mr, total_mr_size: int, device=None) -> DataStruct:
        if device is None:
            device = torch.device("cuda", 0)

        mr_pointer = mr.buf

        # ── 1. Validate magic ─────────────────────────────────────────────────────
        magic_arr = (ctypes.c_uint8 * len(_MAGIC)).from_address(mr_pointer)
        if bytes(magic_arr) != _MAGIC:
            raise ValueError(f"MR magic mismatch — data may be corrupt or incompatible.")
        pos = len(_MAGIC)

        # ── 2. Read JSON length fields ────────────────────────────────────────────
        header_json_len      = struct.unpack("<Q", bytes((ctypes.c_uint8 * 8).from_address(mr_pointer + pos)))[0]
        pos += 8
        tensor_meta_json_len = struct.unpack("<Q", bytes((ctypes.c_uint8 * 8).from_address(mr_pointer + pos)))[0]
        pos += 8

        # ── 3. Parse header JSON ──────────────────────────────────────────────────
        header = json.loads(bytes((ctypes.c_uint8 * header_json_len).from_address(mr_pointer + pos)).decode("utf-8"))
        pos += header_json_len

        # ── 4. Parse tensor-meta JSON ─────────────────────────────────────────────
        tensor_meta = json.loads(bytes((ctypes.c_uint8 * tensor_meta_json_len).from_address(mr_pointer + pos)).decode("utf-8"))
        pos += tensor_meta_json_len

        # ── 5. Skip alignment padding ─────────────────────────────────────────────
        tensor_data_offset = _align_up(pos, _ALIGN)

        # ── 6. Reconstruct nested tensor list (MR → CPU → GPU) ───────────────────
        data = []
        current_offset = tensor_data_offset
        for poly_meta in tensor_meta:
            poly_list = []
            for meta in poly_meta:
                nbytes     = meta["nbytes"]
                shape      = meta["shape"]
                dtype      = getattr(torch, meta["dtype"])
                chunk_addr = mr_pointer + current_offset

                ctypes_arr = (ctypes.c_uint8 * nbytes).from_address(chunk_addr)
                cpu_tensor = (torch.frombuffer(ctypes_arr, dtype=torch.uint8)
                                .view(dtype)
                                .view(shape)
                                .clone())                     # detach from MR memory
                poly_list.append(cpu_tensor.to(device=device, non_blocking=True))
                current_offset += nbytes
            data.append(poly_list)

        torch.cuda.synchronize()
        #print(f"[Deserialize] {sum(len(p) for p in data)} tensor(s) across "
        #      f"{len(data)} poly_list(s) loaded onto {device}.")

        # ── 7. Rebuild DataStruct ─────────────────────────────────────────────────
        ct = DataStruct(
            data             = data,
            include_special  = header["include_special"],
            ntt_state        = header["ntt_state"],
            montgomery_state = header["montgomery_state"],
            origin           = header["origin"],
            level_calc       = header["level_calc"],
            level_available  = header["level_available"],
            hash             = header["hash"],
            version          = header["version"],
        )

        #print(f"[Deserialize] DataStruct AFTER Deserialization(Host - after BS): "
        #      f"level={ct.level}, level_calc={ct.level_calc}, level_avil={ct.level_available}")
        return ct

    def nvme_passthru(self, mr):
        dev_name = "rocep59s0".encode('utf-8')
        dev_name_len = len(dev_name)
        metadata = {
            "rkey": mr.rkey,
            "addr": mr.buf,
            "length": mr.length,
            "devnamelen": dev_name_len,
            "devname": dev_name
        }
        #print(f"rkey: {hex(mr.rkey)}")
        #print(f"addr: {hex(mr.buf)}")
        #print(f"length:{mr.length}")


        cmd = nvme.ndp_passthru_cmd()
        cmd.opcode = 0xdb
        cmd.flags = 0
        cmd.rsvd = 0
        cmd.nsid = 1
        cmd.cdw2 = 0
        cmd.cdw3 = 0
        cmd.cdw10 = 0
        cmd.cdw11 = 0
        cmd.cdw12 = 0
        cmd.cdw13 = 0
        cmd.cdw14 = 0
        cmd.cdw15 = 0
        cmd.data_len = 4096
        cmd.data = 0
        cmd.metadata_len = 0
        cmd.metadata = 0
        cmd.timeout_ms = 60000
        cmd.result = 0

        bufferptr = ctypes.c_void_p()
        buffersize = 4096 #Bytes

        if self.libc.posix_memalign(ctypes.byref(bufferptr), self.libc.getpagesize(),
                            buffersize) != 0:
            raise Exception('ENOMEM')

        ctypes.memset(bufferptr, 0, buffersize)
        cmd.data = bufferptr.value

        tempptr = bufferptr
        pack_format = f"<QQII{dev_name_len}s"
        packed_data = struct.pack(pack_format, metadata['rkey'], metadata['addr'], metadata['length'], metadata['devnamelen'], metadata['devname'])

        ctypes.memmove(bufferptr.value, packed_data, len(packed_data))
        
        result = nvme.ndp_passthru(self.fd, cmd)

        return result

    def receive_bs_result(self, size):
        #rmr = cmid.reg_msgs(size)
        rmr = self.cmid.reg_msgs(52428800)
        self.cmid.post_recv(rmr)
        wc = self.cmid.get_recv_comp()
        if wc is None:
            raise RuntimeError("No recv completion returned")
        print(
                    f"RECV completion: status={wc.status} "
                    f"opcode={wc.opcode} "
                    f"bytes={wc.byte_len}"
                )

        length = 52428800
        return rmr, length


    def bootstrap(self, ct: DataStruct) -> DataStruct:
        """
        Custom bootstrap implementation.

        Replace the body of this method with your own bootstrapping logic.
        The parent's implementation (CkksEngine.bootstrap) is accessible via
        super().bootstrap(ct) if needed as a fallback or reference.

        Args:
            ct (DataStruct): Input ciphertext to be bootstrapped.

        Returns:
            DataStruct: Refreshed ciphertext after bootstrapping.
        """
        # --- Pre-bootstrap GPU cache flush (mirrors parent behaviour) ---
        for device in self.ntt.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        mr, ct_size = self.ct_serialization(ct)
        result = self.nvme_passthru(mr)
        rmr, new_ct_size = self.receive_bs_result(ct_size)
        result_ct = self.ct_deserialization(rmr, new_ct_size)

        # --- Post-bootstrap GPU cache flush (mirrors parent behaviour) ---
        for device in self.ntt.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        return result_ct  # noqa: F821  (defined once TODO block is filled in)