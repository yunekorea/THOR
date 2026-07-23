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

# Which bootstrap-key caching policy to use:
#   "lru"    - LRUBootstrapKeyCache (evict least-recently-used)
#   "belady" - BeladyBootstrapKeyCache (evict whichever key is needed
#              farthest in the future -- optimal, since the access
#              pattern is fully deterministic; see class docstring)
#   "none"   - load all 55 keys straight to GPU, no paging at all
CACHE_POLICY = "belady"

# ----------------------------------------------------------------------
# Memory diagnostics (same helper used in THOR_baseline.py)
# ----------------------------------------------------------------------
# torch.cuda.memory_allocated  -> bytes actually held by live tensors right now
# torch.cuda.memory_reserved   -> bytes PyTorch's caching allocator has carved
#                                  out of the driver (this is what nvidia-smi sees)
# reserved - allocated         -> memory the allocator is sitting on but isn't
#                                  using for a live tensor: fragmentation/cache
def log_mem(tag):
    dev = devices[0]
    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved  = torch.cuda.memory_reserved(dev) / 1024**3
    peak      = torch.cuda.max_memory_allocated(dev) / 1024**3
    gap       = reserved - allocated
    print(f"[MEM] {tag:55s} | alloc={allocated:8.3f}GB | reserved={reserved:8.3f}GB "
          f"| gap={gap:8.3f}GB | peak_alloc={peak:8.3f}GB")

# Counts every serviced bootstrap request, for logging and for handing
# BeladyBootstrapKeyCache its end_of_cycle() signal.
_bootstrap_call_count = [0]

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

    def end_of_cycle(self):
        """No-op for LRU. Exists so the driving code can call
        cache.end_of_cycle() unconditionally regardless of policy —
        BeladyBootstrapKeyCache uses this hook to learn the reference
        access pattern (see that class's docstring)."""
        pass


class BeladyBootstrapKeyCache:
    """
    Belady/MIN (offline-optimal) cache for bootstrap rotation keys.

    Same rationale as on the Host side: THOR's bootstrap circuit issues a
    fixed, data-independent sequence of rotation-key lookups every single
    call — confirmed empirically (a full trace showed 308 consecutive
    bootstraps requesting the identical 114-key sequence, in the identical
    order) and structurally (CKKS's CoeffToSlot/SlotToCoeff/EvalMod steps
    cannot branch on ciphertext content — that would leak the plaintext).
    Because the future access sequence is fully knowable, "evict whichever
    resident key is needed farthest in the future" is provably optimal
    here, not just a heuristic.

    On this Target server, every incoming RDMA request triggers exactly
    one engine.bootstrap() call, so "one cycle" = "one serviced request".
    end_of_cycle() should be called once, right after the very first
    bootstrap() call returns, to lock in the learned pattern; every
    request after that is served under genuine Belady eviction.

    Same no-write-back eviction as LRUBootstrapKeyCache above: rotation
    keys are read-only and _host already holds the untouched, original
    CPU copy from load time, so dropping a GPU entry needs nothing more
    than deleting the reference.
    """

    def __init__(self, engine, host_store: dict, max_gpu_keys: int = 45):
        self._engine = engine
        self._host   = host_store          # permanent CPU store, never mutated
        self._gpu    = OrderedDict()       # {key: DataStruct on GPU}; OrderedDict lets us fall back to LRU while learning
        self._max    = max_gpu_keys

        # Stats for debugging
        self._hits   = 0
        self._misses = 0

        # Pattern-learning state
        self._cycle     = None   # learned reference sequence (list of keys), once known
        self._recording = []     # accumulates every key requested until end_of_cycle() is called
        self._pos        = 0     # index into self._cycle for the *next* lookup

    # ------------------------------------------------------------------
    # Primary interface — bs.bootstrap() calls this as  bs_key[k]
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        if self._cycle is None:
            self._recording.append(key)

        if key in self._gpu:
            self._hits += 1
            if self._cycle is None:
                self._gpu.move_to_end(key)   # keep LRU ordering fresh while still learning
        else:
            self._misses += 1
            if key not in self._host:
                raise KeyError(
                    f"Bootstrap rotation key {key!r} was never loaded. "
                    f"Available keys: {list(self._host.keys())}"
                )

            if len(self._gpu) >= self._max:
                self._evict()

            gpu_ds = self._engine.cuda(self._host[key])
            self._gpu[key] = gpu_ds
            self._gpu.move_to_end(key)

        if self._cycle is not None:
            self._pos = (self._pos + 1) % len(self._cycle)

        return self._gpu[key]

    def end_of_cycle(self):
        """
        Call once, right after the first engine.bootstrap() call returns.
        Locks in the recorded key sequence as the reference cycle that
        every future eviction decision will be based on. A no-op on every
        call after the first.
        """
        if self._cycle is None and self._recording:
            self._cycle = list(self._recording)
            self._recording = None
            self._pos = 0
            print(f"[BeladyBootstrapKeyCache] learned a cycle of length "
                  f"{len(self._cycle)}; switching from LRU fallback to "
                  f"Belady/MIN eviction.")

    def _next_use_distance(self, key, from_pos):
        """Steps from from_pos (inclusive) to key's next occurrence in the
        learned cycle, wrapping around. Every one of the 55 rotation keys
        appears at least once per bootstrap, so this always terminates
        well within len(cycle) steps for any key that came from this
        cache."""
        L = len(self._cycle)
        for step in range(L):
            if self._cycle[(from_pos + step) % L] == key:
                return step
        return L  # unreachable in practice; treat as "farthest possible"

    # ------------------------------------------------------------------
    # Membership / iteration — use CPU store so no GPU side-effects
    # ------------------------------------------------------------------
    def __contains__(self, key):
        return key in self._host

    def __len__(self):
        return len(self._host)

    def keys(self):
        return self._host.keys()

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
    # Belady/MIN eviction
    # ------------------------------------------------------------------
    def _evict(self):
        if self._cycle is None:
            # Haven't learned the pattern yet -- fall back to plain LRU,
            # exactly like LRUBootstrapKeyCache does.
            evict_key, evict_ds = self._gpu.popitem(last=False)
        else:
            # Belady/MIN: evict whichever resident key is needed farthest
            # in the future, looking forward cyclically from the current
            # position in the learned pattern.
            evict_key = max(
                self._gpu.keys(),
                key=lambda k: self._next_use_distance(k, self._pos)
            )
            evict_ds = self._gpu.pop(evict_key)
        # No write-back: _host[evict_key] already holds the untouched
        # original CPU copy from load time (same as LRUBootstrapKeyCache).
        del evict_ds
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
        """Keys currently on the GPU, in LRU → MRU order (meaningless once
        Belady eviction is active, but harmless to keep for inspection)."""
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
            "cycle_learned": self._cycle is not None,
            "cycle_length":  len(self._cycle) if self._cycle is not None else None,
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
    log_mem("after create_cts_stc_const")

    if CACHE_POLICY == "lru":
        cache = LRUBootstrapKeyCache(engine, host_store, max_gpu_keys=50)
        engine.add_bs_key(cache)
    elif CACHE_POLICY == "belady":
        cache = BeladyBootstrapKeyCache(engine, host_store, max_gpu_keys=48)
        engine.add_bs_key(cache)
    elif CACHE_POLICY == "none":
        print("CACHE_POLICY='none' -> loading all 55 rotation keys directly to GPU (no paging)")
        gpu_key_dict = {}
        for k, v in host_store.items():
            gpu_key_dict[k] = engine.cuda(v)
        engine.add_bs_key(gpu_key_dict)
        cache = None
    else:
        raise ValueError(f"Unknown CACHE_POLICY: {CACHE_POLICY!r}")
    log_mem("after rotation keys resident on GPU")
    print(f"Cache policy: {CACHE_POLICY}")
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
    log_mem("after pk/evk/conjk loaded")

    return cache

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

def read_ciphertext(conn, mask, cid, engine, cache):
    data = conn.recv(128)
    if data:
        print("Offloading received! Processing...")
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

            _bootstrap_call_count[0] += 1
            n = _bootstrap_call_count[0]
            log_mem(f"bootstrap #{n:04d} - before")
            try:
                res_ct = engine.bootstrap(ct)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n\n=== OOM on bootstrap request #{n} ===")
                    print(torch.cuda.memory_summary(devices[0], abbreviated=False))
                raise
            log_mem(f"bootstrap #{n:04d} - after ")

            # Lets BeladyBootstrapKeyCache know where one bootstrap's
            # reference sequence ends, so it can lock in the learned
            # cycle. No-op for LRU/none, and a no-op after the first call
            # for Belady too.
            if cache is not None:
                cache.end_of_cycle()
                print(f"Cache stats: {cache.cache_stats}")

            new_mr , new_size = ct_serialization(res_ct, cid)

            print("Post SEND: ", end="")
            cid.post_send(new_mr, new_size)
            wc = cid.get_send_comp()
            if wc is None:
                raise RuntimeError("No send completion returned")
            print("Successful")
            print("Bootstrapping COMPLETE")
            local_mr.close()
            new_mr.close()


        
        except struct.error as e:
            print(f"Error unpacking metadata: {e}")
        except Exception as e:
            print(f"RDMA Operation error: {e}")

    sel.unregister(conn)
    conn.close()

def accept_connection(sock, mask, cid, engine, cache):
    conn, _ = sock.accept()
    conn.setblocking(False)
    # Register the 'read' event for this specific connection
    sel.register(conn, selectors.EVENT_READ,
                data=lambda key_obj, mask_val: read_ciphertext(key_obj.fileobj, mask_val, cid, engine, cache))

def main():
    #key_path = sys.argv[1]
    key_path = "/mnt/nvmf/THOR_test/THOR/keys/keys0"
    cid = RDMA_init()
    engine = engine_init()
    cache = key_init(engine, key_path)
    server = UDS_init()
    sel.register(server, selectors.EVENT_READ,
                 data=lambda key_obj, mask_val: accept_connection(key_obj.fileobj, mask_val, cid, engine, cache))
    print("Target is ready. Waiting for offloading events")
    try:
        while True:
            events = sel.select() # This blocks efficiently (uses epoll/kqueue)
            for key, mask in events:
                callback = key.data
                callback(key, mask)
    except KeyboardInterrupt:
        print("Shutting down...")
        if cache is not None:
            print(f"Final cache stats: {cache.cache_stats}")

if __name__ == '__main__':
    main()
