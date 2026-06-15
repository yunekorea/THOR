import selectors
import socket
import sys
import os
import pprint
import mmap
import struct
import json

import ctypes
from ctypes.util import find_library

from pyverbs.device import Context
from pyverbs.pd import PD
from pyverbs.mr import MR
from pyverbs.libibverbs_enums import ibv_access_flags as fe
from pyverbs.addr import GlobalRoute
from pyverbs.addr import AH, AHAttr
from pyverbs.cmid import CMID, AddrInfo
from pyverbs.qp import QPInitAttr, QPCap, QPAttr, QP
from pyverbs.cq import CQ
from pyverbs.libibverbs_enums import ibv_access_flags, ibv_qp_type, ibv_wr_opcode
from pyverbs.librdmacm_enums import rdma_port_space, RAI_PASSIVE
import pyverbs.wr as pwr

import gc
import collections


project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
project_root = os.path.abspath(os.path.join(os.getcwd(), '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

devices = [0]

import pickle    
import numpy as np
import math
import torch
from transformers import BertForNextSentencePrediction
import matplotlib.pyplot as plt

from liberate.fhe.bootstrapping import ckks_bootstrapping as bs
from liberate.fhe.data_struct import DataStruct

import thor
from thor import CkksEngine, ThorDataEncryptor, ThorLinearEvaluator
from thor.bert import ThorBert, ThorBertFF, ThorBertPooler, ThorBertClassifier

import time

sel = selectors.DefaultSelector()

from collections import OrderedDict


class LRUBootstrapKeyCache:
    """
    Transparent dict-like wrapper for the bs_key (rotk_dict) that keeps at
    most `max_gpu_keys` rotation keys on the GPU at once.

    All 55 keys are permanently stored on the CPU in `_host`.
    The GPU cache (`_gpu`) is bounded by `max_gpu_keys` and managed with LRU.

    When bs.bootstrap() does  bs_key[k]:
      - Hit  : return the GPU tensor, mark as MRU.
      - Miss : evict the LRU key if full (just delete it — CPU copy is
               already safe in _host), then upload the needed key to GPU.
    """

    def __init__(self, engine, host_store: dict, max_gpu_keys: int = 4):
        """
        Parameters
        ----------
        engine        : CKKS engine  (provides .cuda() and .cpu())
        host_store    : {rotation_key: DataStruct on CPU}  — owns the data
        max_gpu_keys  : GPU cache capacity (tune to fit your VRAM budget)
        """
        self._engine = engine
        self._host   = host_store     # permanent CPU store, never mutated
        self._gpu    = OrderedDict()  # {key: DataStruct on GPU}, LRU order
        self._max    = max_gpu_keys

        # Stats for debugging
        self._hits   = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Primary interface — bs.bootstrap() calls this as  bs_key[k]
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        if key in self._gpu:
            self._gpu.move_to_end(key)   # promote to MRU
            self._hits += 1
            return self._gpu[key]

        # Cache miss
        self._misses += 1

        if key not in self._host:
            raise KeyError(
                f"Bootstrap rotation key {key!r} was never loaded. "
                f"Available keys: {list(self._host.keys())}"
            )

        # Evict LRU entry if at capacity
        if len(self._gpu) >= self._max:
            self._evict_lru()

        # Upload CPU → GPU and cache it
        gpu_ds = self._engine.cuda(self._host[key])
        self._gpu[key] = gpu_ds
        self._gpu.move_to_end(key)       # mark as MRU
        return gpu_ds

    # ------------------------------------------------------------------
    # Membership / iteration — use CPU store so no GPU side-effects
    # ------------------------------------------------------------------
    def __contains__(self, key):
        return key in self._host

    def __len__(self):
        return len(self._host)

    def keys(self):
        return self._host.keys()

    # Prevent accidental full-cache GPU upload
    def values(self):
        raise NotImplementedError(
            "Iterating .values() would upload all 55 keys to GPU. "
            "Access individual keys via bs_key[k] instead."
        )

    def items(self):
        raise NotImplementedError(
            "Iterating .items() would upload all 55 keys to GPU. "
            "Access individual keys via bs_key[k] instead."
        )

    # ------------------------------------------------------------------
    # LRU eviction
    # ------------------------------------------------------------------
    def _evict_lru(self):
        """
        Drop the least-recently-used GPU DataStruct.
        The CPU copy in self._host is untouched — it was loaded from disk
        with move_to_gpu=False and never moved, so nothing needs writing back.
        """
        lru_key, lru_ds = self._gpu.popitem(last=False)  # last=False → LRU end
        del lru_ds
        gc.collect()
        torch.cuda.empty_cache()

    def evict_all(self):
        """Flush the entire GPU cache. Call this after bootstrapping is done."""
        keys = list(self._gpu.keys())
        for k in keys:
            del self._gpu[k]
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def gpu_resident_keys(self) -> list:
        """Keys currently on the GPU, in LRU → MRU order."""
        return list(self._gpu.keys())

    @property
    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        return {
            "gpu_resident":  len(self._gpu),
            "max_gpu":       self._max,
            "total_keys":    len(self._host),
            "hits":          self._hits,
            "misses":        self._misses,
            "hit_rate":      f"{hit_rate:.1%}",
        }

# Magic bytes to catch corruption / version mismatch
_MAGIC = b"LBFHE001"
_ALIGN  = 64   # align tensor data to 64-byte boundary (cache-line friendly)


def _align_up(n: int, align: int) -> int:
    return int(math.ceil(n / align) * align)

def ct_serialization(ct: DataStruct, cmid):
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
    #print(f"[Serialize] DataStruct BEFORE Serialization(Target - after BS): "
    #      f"level={ct.level}, level_calc={ct.level_calc}, level_avail={ct.level_available}")

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

    #print(f"Target MR layout(After BS): {preamble_size}B preamble + "
    #      f"{tensor_data_offset - preamble_size}B padding + "
    #      f"{total_tensor_bytes}B tensor payload = {total_mr_size}B total")

    # ── 4. Allocate RDMA MR ───────────────────────────────────────────────────
    mr         = cmid.reg_msgs(total_mr_size)
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

def ct_deserialization(mr, total_mr_size: int, device=None) -> DataStruct:
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

    #print(f"[Deserialize] DataStruct AFTER Deserialization(Target - before BS): "
    #      f"level={ct.level}, level_calc={ct.level_calc}, level_avail={ct.level_available}")
    
    return ct

def engine_init():
    print("engine init: ", end="")
    with torch.cuda.device(devices[0]):
        torch.cuda.empty_cache()
    params = {"logN":16, "scale_bits": 41, "num_special_primes": 4, "devices": devices, "quantum":"pre_quantum"}
    engine = CkksEngine(params)
    print("DONE")
    return engine

def key_init(engine, key_path):
    print("key init:")
    rotk_dict_keys = [
        -32768, -16384, -1024, -512, -32, -16,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384,
        416, 448, 480, 512, 1024, 2048, 3072, 4096, 5120, 6144,
        7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336,
        15360, 16384
    ]
    # Force an immediate clean start
    gc.collect()
    torch.cuda.empty_cache()

    host_store = {}
    numkeys = len(rotk_dict_keys)
    
    print("ROTK dict(Btstrp): ", end="")
    for i, key in enumerate(rotk_dict_keys, 1):
        host_store[key] = engine.load(
            f"{key_path}/rotk_dict/{key}",
            move_to_gpu=False          # stays on CPU DRAM
        )
        #print(f"loaded keys (CPU): {i}/{numkeys}")

    bs.create_cts_stc_const(engine)

    lru_cache = LRUBootstrapKeyCache(engine, host_store,
                                     max_gpu_keys=35)
    engine.add_bs_key(lru_cache)
    print("DONE")

    gc.collect()
    torch.cuda.empty_cache()

    print("pk: ", end="")
    pk = engine.load(f"{key_path}/pk")
    engine.add_pk(pk)
    del pk
    gc.collect()
    print("DONE")
    print("evk: ", end="")
    evk = engine.load(f"{key_path}/evk")
    engine.add_evk(evk)
    del evk
    gc.collect()
    print("DONE")
    torch.cuda.empty_cache()
    time.sleep(10)
    print("conjk: ", end="")
    conjk = engine.load(f"{key_path}/conjk")
    engine.add_conj_key(conjk)
    del conjk
    gc.collect()
    print("DONE")

def RDMA_init():
    dev_name = "mlx5_0".encode('utf-8')
    dev_name_len = len(dev_name)
    ctx = Context(name='mlx5_0')
    
    cap = QPCap(max_send_wr=16, max_recv_wr=16, max_send_sge=8)
    qp_init_attr = QPInitAttr(cap=cap)

    # Initialize CIMD
    host_ip = "192.168.100.2"
    target_ip = "192.168.100.1"
    #cai = AddrInfo(src = target_ip,dst=host_ip, dst_service="9999",
    #                port_space = rdma_port_space.RDMA_PS_TCP)

    cai = AddrInfo(src=target_ip, src_service="9999",
                port_space = rdma_port_space.RDMA_PS_TCP, flags = RAI_PASSIVE)
    cid = CMID(creator=cai, qp_init_attr=qp_init_attr)
    print("cid listen")
    cid.listen()
    new_id = cid.get_request()
    new_id.accept()
    return new_id

def UDS_init():
    print("UDS init: ", end="")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists('/tmp/rdma_metadata.sock'):
        os.remove('/tmp/rdma_metadata.sock')
    server.bind('/tmp/rdma_metadata.sock')
    server.listen(1)
    server.setblocking(False)
    print("DONE")
    return server

def read_ciphertext(conn, mask, cid, engine):
    data = conn.recv(128)
    if data:
        print("Offloading received! Processing metadata...")
        # Process your RDMA logic here
        struct_format = "<QQII50s" 
        
        try:
            # We slice the data to match the expected struct size
            unpacked = struct.unpack(struct_format, data[:struct.calcsize(struct_format)])
            
            rkey        = unpacked[0]
            addr        = unpacked[1]
            length      = unpacked[2]
            name_length = unpacked[3]
            # Decode the name and strip null bytes (\x00)
            device_name = unpacked[4][:name_length].decode('utf-8').strip('\x00')
            
            #print(f"--- Decoded Metadata ---")
            #print(f"Address:     {hex(addr)}")
            #print(f"R-Key:       {hex(rkey)}")
            #print(f"Length:      {length}")
            #print(f"Name Length: {name_length}")
            #print(f"Device Name: {device_name}")
            
            # Now you can proceed with your RDMA logic using these variables
            #cid.connect()
        
            local_mr = cid.reg_msgs(length)
            #print("MR set")
            cid.post_read(local_mr, length, addr, rkey)
            wc = cid.get_send_comp()
            if wc is None:
                raise RuntimeError("No READ completion returned")

            #print(
            #    f"READ completion: status={wc.status} "
            #    f"opcode={wc.opcode} "
            #    f"bytes={wc.byte_len}"
            #)

            ct = ct_deserialization(local_mr, length)
            res_ct = engine.bootstrap(ct)
            new_mr , new_size = ct_serialization(res_ct, cid)

            print("Post SEND: ", end="")
            cid.post_send(new_mr, new_size)
            wc = cid.get_send_comp()
            if wc is None:
                raise RuntimeError("No send completion returned")
            print("Successful")
            local_mr.close()
            new_mr.close()


        
        except struct.error as e:
            print(f"Error unpacking metadata: {e}")
        except Exception as e:
            print(f"RDMA Operation error: {e}")

    sel.unregister(conn)
    conn.close()

def accept_connection(sock, mask, cid, engine):
    conn, _ = sock.accept()
    conn.setblocking(False)
    # Register the 'read' event for this specific connection
    sel.register(conn, selectors.EVENT_READ,
                data=lambda key_obj, mask_val: read_ciphertext(key_obj.fileobj, mask_val, cid, engine))

def main():
    #key_path = sys.argv[1]
    key_path = "/mnt/nvmf/THOR_test/THOR/keys/keys0"
    cid = RDMA_init()
    engine = engine_init()
    key_init(engine, key_path)
    server = UDS_init()
    sel.register(server, selectors.EVENT_READ,
                 data=lambda key_obj, mask_val: accept_connection(key_obj.fileobj, mask_val, cid, engine))
    print("Target is ready. Waiting for offloading events")
    try:
        while True:
            events = sel.select() # This blocks efficiently (uses epoll/kqueue)
            for key, mask in events:
                callback = key.data
                callback(key, mask)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == '__main__':
    main()
