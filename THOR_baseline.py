import sys
import os 

project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
project_root = os.path.abspath(os.path.join(os.getcwd(), '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

import pickle    
import numpy as np
import math
import torch
from transformers import BertForNextSentencePrediction
import matplotlib.pyplot as plt

from liberate.fhe.bootstrapping import ckks_bootstrapping as bs

import thor
from thor import CkksEngine, ThorDataEncryptor, ThorLinearEvaluator
from thor.bert import ThorBert, ThorBertFF, ThorBertPooler, ThorBertClassifier
from liberate.fhe.data_struct import DataStruct
from thor.ckks_ndp import CkksNDPEngine

#from thor import CkksNDPEngine

#Modules for NDP, RDMA
import pprint
import mmap
import struct
import json

from collections import OrderedDict
import gc

from codetiming import Timer as timer


devices = [0]

dataset_type = 'mrpc'
target_idx = 0

rotk_dict_keys = [
    -32768, -16384, -1024, -512, -32, -16,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384,
    416, 448, 480, 512, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336,
    15360, 16384
]                    

deltas = [    
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 16, 2048, 4096, 6144, 8192, 10240,
    12288, 14336, 18432, 20480, 22528, 24576, 26624, 28672, 30720,
    256, 2288, 4320, 6352, 8384, 10416, 12448, 14480, 16512,
    18544, 20576, 22608, 24640, 26672, 28704, 30736,
    512, 768, 1280, 2544, 2800, 3312, 4576, 4832, 5344,
    6608, 6864, 8640, 8896, 10672,
]

dataset = f'./datasets/{dataset_type}'

variables_list = []
h_indices = [np.where(np.arange(0, 2**11) % 16 == i) for i in range(12)]

key_timer = timer(name = "key_timer")

class LRUBootstrapKeyCache:
    """
    A dict-like wrapper that keeps at most `max_gpu_keys` bootstrap rotation
    keys on the GPU at once.  All 55 keys live on the CPU (host_store); the
    GPU cache is managed with an LRU policy.

    The bootstrapping code does  bs_key[rotation_index]  — this class
    intercepts that lookup, moves the key to GPU on demand, and evicts the
    least-recently-used key back to CPU when the cache is full.

    Usage
    -----
        cache = LRUBootstrapKeyCache(engine, host_store, max_gpu_keys=4)
        engine.add_bs_key(cache)           # replaces the old rotk_dict
    """

    def __init__(self, engine, host_store: dict, max_gpu_keys: int = 4):
        """
        Parameters
        ----------
        engine        : the CKKS engine (needs a .cuda() method for host→GPU)
        host_store    : dict  {rotation_key: DataStruct on CPU}
        max_gpu_keys  : how many keys to keep resident on the GPU at once.
                        Keep this small enough to avoid OOM.
        """
        self._engine = engine
        self._host   = host_store          # CPU copies, never evicted
        self._gpu    = OrderedDict()       # GPU copies, LRU-ordered
        self._max    = max_gpu_keys

        # Hit-ratio bookkeeping
        self._hits   = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Core lookup – called as  bs_key[k]  by the bootstrapping internals
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        print(f"Called KEY: {key}")
        if key in self._gpu:
            # Cache hit → move to "most recently used" end
            self._hits += 1
            self._gpu.move_to_end(key)
            return self._gpu[key]

        # Cache miss → load from CPU
        self._misses += 1
        if key not in self._host:
            raise KeyError(f"Bootstrap key {key!r} not found in host store.")

        # Evict LRU key if we are at capacity
        if len(self._gpu) >= self._max:
            self._evict_lru()

        # Move key from CPU → GPU
        gpu_key = self._engine.cuda(self._host[key])
        self._gpu[key] = gpu_key
        self._gpu.move_to_end(key)         # mark as MRU
        return gpu_key

    # ------------------------------------------------------------------
    # Pass-through helpers so the bootstrapping code can iterate / test
    # membership without triggering GPU loads
    # ------------------------------------------------------------------
    def __contains__(self, key):
        return key in self._host           # logical membership = all keys

    def __len__(self):
        return len(self._host)

    def keys(self):
        return self._host.keys()

    def values(self):
        # Iterating values would page everything onto the GPU – warn loudly.
        raise NotImplementedError(
            "Iterating .values() would move all keys to GPU. "
            "Use explicit key lookups instead."
        )

    def items(self):
        raise NotImplementedError(
            "Iterating .items() would move all keys to GPU. "
            "Use explicit key lookups instead."
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def _evict_lru(self):
        """Move the least-recently-used GPU key back to CPU and free VRAM."""
        lru_key, lru_tensor = self._gpu.popitem(last=False)  # FIFO end = LRU
        # Move the DataStruct's tensors back to CPU in-place
        self._host[lru_key] = self._engine.cpu(lru_tensor)
        del lru_tensor
        gc.collect()
        torch.cuda.empty_cache()

    def evict_all(self):
        """Push every cached GPU key back to CPU.  Call after bootstrapping."""
        while self._gpu:
            self._evict_lru()

    @property
    def gpu_resident_keys(self):
        """Which keys are currently on the GPU (for debugging)."""
        return list(self._gpu.keys())

    @property
    def cache_stats(self):
        return {
            "gpu_resident": len(self._gpu),
            "max_gpu":      self._max,
            "total_keys":   len(self._host),
        }

    @property
    def hit_ratio(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def end_of_cycle(self):
        """No-op for LRU. Exists so the driving code can call
        cache.end_of_cycle() unconditionally regardless of which policy
        is active (BeladyBootstrapKeyCache uses this hook to learn the
        reference access pattern)."""
        pass


class BeladyBootstrapKeyCache:
    """
    Belady/MIN (offline-optimal) cache for bootstrap rotation keys.

    Why this applies here: THOR's bootstrap circuit is a fixed,
    data-independent sequence of rotation-key lookups. This was confirmed
    two ways earlier in this investigation -- empirically (308 consecutive
    bootstraps in a full trace all requested the identical 114-key
    sequence, in the identical order) and structurally (CKKS's
    CoeffToSlot/SlotToCoeff/EvalMod steps cannot branch on ciphertext
    content -- doing so would leak the plaintext). Because the future
    access sequence is fully knowable, "evict whichever resident key is
    needed farthest in the future" is not a heuristic here -- it is the
    provably optimal eviction policy for a fixed cache size (Belady 1966).

    For context, measured on this workload: plain LRU gets a 68.3% hit
    rate. With 55 distinct keys and a 45-key cap, no policy can ever do
    better than 10 misses per 114 lookups (~91.2% hit rate) -- that's the
    structural floor, since 10 keys must always be absent. Belady, once it
    has learned the pattern, converges to exactly that floor.

    Design: rather than hardcoding the cycle length (fragile if the
    bootstrap parameters ever change), this cache *learns* it at runtime.
    The very first bootstrap is served under a plain LRU fallback while
    every lookup is recorded. `end_of_cycle()` -- called once, right after
    that first engine.bootstrap() call returns -- locks in the recorded
    sequence as the reference cycle. From then on, every lookup advances a
    position pointer into that cycle, and every eviction picks whichever
    resident key's next occurrence (scanned forward cyclically from the
    current position) is farthest away.
    """

    def __init__(self, engine, host_store: dict, max_gpu_keys: int = 45):
        self._engine = engine
        self._host   = host_store          # CPU copies, never evicted
        self._gpu    = OrderedDict()       # GPU copies; OrderedDict lets us fall back to LRU while learning
        self._max    = max_gpu_keys

        # Hit-ratio bookkeeping
        self._hits   = 0
        self._misses = 0

        # Pattern-learning state
        self._cycle     = None   # learned reference sequence (list of keys), once known
        self._recording = []     # accumulates every key requested until end_of_cycle() is called
        self._pos        = 0     # index into self._cycle for the *next* lookup

    # ------------------------------------------------------------------
    # Core lookup – called as  bs_key[k]  by the bootstrapping internals
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
                raise KeyError(f"Bootstrap key {key!r} not found in host store.")

            if len(self._gpu) >= self._max:
                self._evict()

            gpu_key = self._engine.cuda(self._host[key])
            self._gpu[key] = gpu_key
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
    # Cache management
    # ------------------------------------------------------------------
    def _evict(self):
        if self._cycle is None:
            # Haven't learned the pattern yet -- fall back to plain LRU,
            # exactly like LRUBootstrapKeyCache does.
            evict_key, evict_tensor = self._gpu.popitem(last=False)
        else:
            # Belady/MIN: evict whichever resident key is needed farthest
            # in the future, looking forward cyclically from the current
            # position in the learned pattern.
            evict_key = max(
                self._gpu.keys(),
                key=lambda k: self._next_use_distance(k, self._pos)
            )
            evict_tensor = self._gpu.pop(evict_key)

        self._host[evict_key] = self._engine.cpu(evict_tensor)
        del evict_tensor
        gc.collect()
        torch.cuda.empty_cache()

    def evict_all(self):
        """Push every cached GPU key back to CPU. Call after bootstrapping."""
        while self._gpu:
            self._evict()

    # ------------------------------------------------------------------
    # Pass-through helpers, matching LRUBootstrapKeyCache's interface
    # ------------------------------------------------------------------
    def __contains__(self, key):
        return key in self._host

    def __len__(self):
        return len(self._host)

    def keys(self):
        return self._host.keys()

    def values(self):
        raise NotImplementedError(
            "Iterating .values() would move all keys to GPU. "
            "Use explicit key lookups instead."
        )

    def items(self):
        raise NotImplementedError(
            "Iterating .items() would move all keys to GPU. "
            "Use explicit key lookups instead."
        )

    @property
    def gpu_resident_keys(self):
        return list(self._gpu.keys())

    @property
    def cache_stats(self):
        return {
            "gpu_resident":  len(self._gpu),
            "max_gpu":       self._max,
            "total_keys":    len(self._host),
            "cycle_learned": self._cycle is not None,
            "cycle_length":  len(self._cycle) if self._cycle is not None else None,
        }

    @property
    def hit_ratio(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


def encode_attention_mask(engine, attention_mask:np.ndarray, level:int=15) -> np.ndarray:
    """
    Return an array of size (8,) which contains 8 plaintexts. 
    """
    if attention_mask.shape != (128,):
        raise ValueError("Shape of attention mask should be (128,)")
    n_tokens = np.count_nonzero(attention_mask)
    attention_mask = np.full((8,), None, dtype=object)
    for i in range(8):
        msg = np.zeros((2**15,), dtype=float)
        for j in range(16):
            temp = j *(2**11)
            diag_index = i * 16 + j
            for t in range(128):
                col_index = (diag_index + t) % 128
                is_token = 1 if col_index < n_tokens else 0
                for head in range(12):
                    msg[temp + t*16 + head] = is_token
        attention_mask[i] = engine.encode(msg, level)
    return attention_mask


with torch.cuda.device(devices[0]):
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(devices[0]) /1024**3)

# ----------------------------------------------------------------------
# Memory diagnostics
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

# Which bootstrap-key caching policy to use:
#   "lru"    - LRUBootstrapKeyCache (evict least-recently-used)
#   "belady" - BeladyBootstrapKeyCache (evict whichever key is needed
#              farthest in the future -- optimal, since the access
#              pattern is fully deterministic; see class docstring)
#   "none"   - load all 55 keys straight to GPU, no paging at all
CACHE_POLICY = "belady"

params = {"logN":16, "scale_bits": 41, "num_special_primes": 4, "devices": devices, "quantum":"pre_quantum"}
engine = CkksEngine(params)
print("Memory allocated: ", torch.cuda.memory_allocated(devices[0]) /1024**3)


print("Key Loading: ", end="")
key_timer.start()
#sk = engine.load("./keys/keys0/sk")
pk = engine.load(f"./keys/keys0/pk")
engine.add_pk(pk)
evk = engine.load(f"./keys/keys0/evk")
engine.add_evk(evk)
gk = engine.load(f"./keys/keys0/gk")
engine.add_gk(gk)
conjk = engine.load(f"./keys/keys0/conjk")
engine.add_conj_key(conjk)

host_store = {}
numkeys = len(rotk_dict_keys)

for i, key in enumerate(rotk_dict_keys, 1):
    host_store[key] = engine.load(
        f"./keys/keys0/rotk_dict/{key}",
        move_to_gpu=False          # stays on CPU DRAM
    )
    #print(f"loaded keys (CPU): {i}/{numkeys}")

bs.create_cts_stc_const(engine)
log_mem("after create_cts_stc_const")

if CACHE_POLICY == "lru":
    cache = LRUBootstrapKeyCache(engine, host_store, max_gpu_keys=45)
    engine.add_bs_key(cache)
elif CACHE_POLICY == "belady":
    cache = BeladyBootstrapKeyCache(engine, host_store, max_gpu_keys=45)
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
#engine.add_rot_keys_from_sk(deltas, sk)
key_timer.stop()
print("DONE")
print("Memory allocated: ", torch.cuda.memory_allocated(devices[0]) /1024**3)

# ----------------------------------------------------------------------
# Instrument every bootstrap() call, including ones invoked internally by
# he_softmax / he_gelu / he_layernorm — not just the 18 explicit call sites
# visible in forward_layer. This is a transparent wrapper: behavior is
# unchanged, it just logs memory immediately before and after each call.
# ----------------------------------------------------------------------
_bootstrap_call_count = [0]
_original_bootstrap = engine.bootstrap
def _instrumented_bootstrap(ct):
    _bootstrap_call_count[0] += 1
    n = _bootstrap_call_count[0]
    log_mem(f"bootstrap #{n:04d} - before")
    result = _original_bootstrap(ct)
    log_mem(f"bootstrap #{n:04d} - after ")
    # Lets BeladyBootstrapKeyCache know where one bootstrap's reference
    # sequence ends, so it can lock in the learned cycle. No-op for LRU
    # and a no-op after the first call for Belady too.
    if cache is not None:
        cache.end_of_cycle()
    return result
engine.bootstrap = _instrumented_bootstrap


data_encryptor = ThorDataEncryptor(dataset_type, dataset,
                                   embedding_model=BertForNextSentencePrediction.from_pretrained('bert-base-uncased').bert.embeddings, 
                                   ckks_engine=engine, test=False)
data_loader = data_encryptor.eval_dataloader

idx = 0
for batch in data_loader:
    if idx < target_idx:
        idx += 1
        continue
    if idx == target_idx:
        data= {k: v for k, v in batch.items() if k in ['input_ids', 'token_type_ids']}
        embedding = data_encryptor.embed_data(data)
        x = data_encryptor.encrypt_embedding(embedding, pk, level = 20)
        attention_mask = batch['attention_mask']
        thor_attention_mask = data_encryptor.encode_attention_mask(attention_mask.cpu().numpy().squeeze().T, level=15)
        break


print("Load Model Weights: ", end="")
with open(f"./encoded_models_new/{dataset_type}/att.pkl", 'rb') as f:
    weights_pt = pickle.load(f)

with open(f"./encoded_models_new/{dataset_type}/ff.pkl", 'rb') as f:
    ff_weights = pickle.load(f)
    
with open(f"./encoded_models_new/{dataset_type}/pooler.pkl", 'rb') as f:
    pooler_weights = pickle.load(f)

with open(f"./encoded_models_new/{dataset_type}/cls.pkl", 'rb') as f:
    classifier_weights = pickle.load(f)
print("DONE")

print("Initiate HE Model: ", end="")
evaluator = ThorLinearEvaluator(engine) #LinearEvaluator does operations such as HE-matmul.

thor_bert = ThorBert(evaluator, weights_pt)
thor_ffs = []

for i in range(12):
    thor_ffs.append(ThorBertFF(evaluator, ff_weights, i))
thor_bert.ffs = thor_ffs
thor_bert.pooler = ThorBertPooler(evaluator, pooler_weights)
thor_bert.classifier = ThorBertClassifier(evaluator, classifier_weights)
print("DONE")

ct_test = None
def forward_layer(x, layer_idx, thor_ff):
    global engine, evaluator, thor_attention, thor_attention_mask, time1, time2, time3, time4, time5, time6, time7, time8, time9, time10, time11, time12, time13, time14
    
    thor_attention.to(devices)
    thor_ff.to(devices)
    print("layer_idx:", layer_idx)
    #log_mem(f"layer {layer_idx:02d} - after weights .to(gpu)")
    
    if x.shape == (8,):
        x_cplx = np.full((4,), None, dtype=object)
        for i in range(4):
            x_cplx[i] = engine.cc_add(x[i], engine.imult(x[i+4]))
        if layer_idx != 0:
            for i in range(4):
                x_cplx[i] = engine.cc_add(x_cplx[i], engine.rotate_left(x_cplx[i], -6))
    elif x.shape == (4,):
        x_cplx = x
        x = np.full((8,), None, dtype=object)
        for i in range(4):
            conj = engine.conjugate(x_cplx[i])
            x[i] =  engine.mult_scalar(engine.cc_add(x_cplx[i], conj), 1/2)
            x[i+4] =  engine.mult_scalar(engine.imult(engine.cc_sub(conj, x_cplx[i])), 1/2)
            x_cplx[i] = engine.level_up(x_cplx[i], 21)
    #log_mem(f"layer {layer_idx:02d} - WF0 x_cplx built")
    
    # WF 1. Attention layer
    x_cplx_rots = evaluator.make_rotated_copies(x_cplx)
    #log_mem(f"layer {layer_idx:02d} - WF1a rotated copies made")
    q_wo_rescale = thor_attention.query(x_cplx_rots)
    k = thor_attention.key(x_cplx_rots)
    v = thor_attention.value(x_cplx_rots)
    #log_mem(f"layer {layer_idx:02d} - WF1b q/k/v computed")

    l_k = evaluator.transpose_upper_to_lower(k)
    l_k_cplx = np.full((4,), None, dtype=object)
    for i in range(4):
        l_k_cplx[i] = engine.cc_add(engine.level_up(l_k[i], l_k[i].level_calc+1), engine.imult(evaluator.rotate_internal(l_k[i], 64, mode='att')))
        l_k_cplx[i] = engine.rescale(l_k_cplx[i])
    #log_mem(f"layer {layer_idx:02d} - WF1c transpose+complexify l_k")
    
    # WF 2. Attention score
    q = np.full_like(q_wo_rescale, None, dtype=object)
    for i in range(4):
        q[i] = engine.rescale(q_wo_rescale[i])
    q_copies = evaluator.make_copies(q)
    sftmx_scale = 1
    sftmx_in = thor_attention.calculate_attention_score(l_k_cplx, q_copies, bootstrap=False, scale=sftmx_scale, rescale=False)

    for i in range(4):
        temp = engine.cc_add(sftmx_in[i], engine.imult(sftmx_in[i+4]))
        # Bootstrap #1
        temp = engine.bootstrap(temp)
        conj = engine.conjugate(temp)
        sftmx_in[i] = engine.cc_add(temp, conj)
        sftmx_in[i+4] = engine.imult(engine.cc_sub(conj, temp))
    #log_mem(f"layer {layer_idx:02d} - WF2 attention score + bootstrap#1 block done")
    
    # WF 3. Soft weights(Softmax)
    sftmx_out = thor_attention.softmax(x=sftmx_in, attention_mask=thor_attention_mask, rescale=False, debug=False, sk=None)
    #log_mem(f"layer {layer_idx:02d} - WF3 softmax done")

    v_cplx = np.full((2,), None, dtype=object)
    for i in range(2):
        v_cplx[i] = engine.cc_add(v[i], engine.imult(v[i+2]))
    if sftmx_out[0].level_calc < v_cplx[0].level_calc:
        for j in range(128):
            sftmx_out[j] = engine.level_up(sftmx_out[j], v[0].level_calc)
    elif sftmx_out[0].level_calc > v_cplx[0].level_calc:
        for j in range(2):
            v_cplx[j] = engine.level_up(v_cplx[j], sftmx_out[0].level_calc)
    for i in range(2):
        v_cplx[i] = engine.rescale(v_cplx[i])
    sftmx_out_rescale = np.full((128,), None, dtype=object)
    for j in range(128):
        sftmx_out_rescale[j] = engine.rescale(sftmx_out[j])
    #log_mem(f"layer {layer_idx:02d} - WF3b level-matched + rescaled (128-elem sftmx_out_rescale live)")
    # WF 4. Attention head & multi-head attention
    att_context = thor_attention.calculate_attention_context(v_cplx, sftmx_out_rescale, rescale=False)
    #log_mem(f"layer {layer_idx:02d} - WF4a attention context computed")

    # Bootstrap #2
    for i in range(2):
        att_context[i] = engine.bootstrap(att_context[i])
    #log_mem(f"layer {layer_idx:02d} - WF4b bootstrap#2 done")

    att_context_rots = thor_attention.evaluator.make_rotated_copies(att_context)
    dense_output = thor_attention.dense(att_context_rots)
    x_out_sum = np.full((8,), None, dtype=object)
    mask = np.array(([1]*6+[0]*10)*2**11)
    for i in range(4):
        x_out_sum[i] = engine.add(x[i], dense_output[i])
        x_out_sum[i+4] = engine.add(x[i+4], dense_output[i+4])
    ln1_in = x_out_sum
    #log_mem(f"layer {layer_idx:02d} - WF4c dense output + residual add done")

    # WF 5. LayerNorm1
    ln1_out = thor_attention.layernorm(x=ln1_in, sk=None)
    #log_mem(f"layer {layer_idx:02d} - WF5 layernorm1 done")
    l = np.full((64,), None,dtype=object)
    mask = np.full((engine.num_slots,), 1, dtype=int)
    mask[np.arange(engine.num_slots) % (16) >= 6] = 0
    for i in range(4):
        temp = engine.cc_add(ln1_out[i], engine.imult(ln1_out[i+4]))
        temp = engine.mc_mult(mask, temp)
        l[16*i] = engine.cc_add(temp, engine.rotate_left(temp, -8))
        for j in range(1, 16):
            index = 16*i+j
            l[index] = engine.rotate_left(l[index-1], 2**11)
    #log_mem(f"layer {layer_idx:02d} - WF5b 64-elem 'l' array built")

    # WF 6. FC1(Fully Connected layer; Feed-Forward Network Part 1)
    gelu_in_wo_bs = thor_ff.dense1(l)
    #log_mem(f"layer {layer_idx:02d} - WF6a FC1 dense1 done")

    for i in range(8):
        temp = engine.cc_add(gelu_in_wo_bs[0,i], engine.imult(gelu_in_wo_bs[1,i]))
        temp = engine.mult_scalar(temp, 1/2)
        #Bootstrap #3
        temp = engine.bootstrap(temp)
        conj = engine.conjugate(temp)
        gelu_in_wo_bs[0,i] = engine.cc_add(temp, conj)
        gelu_in_wo_bs[1,i] = engine.imult(engine.cc_sub(conj, temp))
    #log_mem(f"layer {layer_idx:02d} - WF6b bootstrap#3 block done (8 calls)")

    # WF 7. GELU(Activation Function)
    gelu_out = thor_ff.gelu(x=gelu_in_wo_bs)
    #log_mem(f"layer {layer_idx:02d} - WF7 gelu done")
    # WF 8. FC2(Feed-Forward Network Part 2) 
    dense2_out = thor_ff.dense2(gelu_out)
    ln2_in = np.full((8,), None, dtype=object)
    for i in range(8):
        ln2_in[i] = engine.add(ln1_out[i], dense2_out[i])
    for i in range(4):
        temp = engine.cc_add(ln2_in[i], engine.imult(ln2_in[i+4]))
        # Bootstrap #4
        temp = engine.bootstrap(temp)
        conj = engine.conjugate(temp)
        ln2_in[i] = engine.cc_add(temp, conj)
        ln2_in[i+4] = engine.imult(engine.cc_sub(conj, temp)) 
    #log_mem(f"layer {layer_idx:02d} - WF8 FC2 + bootstrap#4 block done")

    # WF 9. LayerNorm2
    if layer_idx == 9 or layer_idx == 10:
        ln2_out = thor_ff.layernorm(x=ln2_in, sk=None)
    else:
        ln2_out = thor_ff.layernorm(x=ln2_in, sk=None)
    #log_mem(f"layer {layer_idx:02d} - WF9 layernorm2 done")

    if ln2_out[0].level >8:
        for i in range(8):
            ln2_out[i] = engine.level_up(ln2_out[i], 21)
        
    thor_attention.cpu()
    thor_ff.cpu()
    #log_mem(f"layer {layer_idx:02d} - after weights .cpu() (layer end)")
    return ln2_out, (x, q_wo_rescale, sftmx_in, sftmx_out, att_context, ln1_in, ln1_out, gelu_in_wo_bs, gelu_out, dense2_out, ln2_in, ln2_out)

print("BASELINE test")

for layer_idx in range(12):
    print(f"Forwarding layer #{layer_idx}: ", end="")
    torch.cuda.reset_peak_memory_stats(devices[0])
    log_mem(f"layer {layer_idx:02d} - loop start (before .to())")

    thor_attention = thor_bert.attentions[layer_idx]
    thor_ff = thor_bert.ffs[layer_idx]

    try:
        x, variables = forward_layer(x, layer_idx, thor_ff)  # ← update x each time
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n\n=== OOM at layer {layer_idx} ===")
            print(torch.cuda.memory_summary(devices[0], abbreviated=False))
        raise

    log_mem(f"layer {layer_idx:02d} - after forward_layer return ('variables' still referenced)")

    # DIAGNOSTIC: 'variables' (the big tuple of this layer's intermediate
    # ciphertexts: x, q_wo_rescale, sftmx_in, sftmx_out (128 elems!),
    # att_context, ln1_in, ln1_out, gelu_in_wo_bs, gelu_out, dense2_out,
    # ln2_in, ln2_out) is never read anywhere in this loop. But because it's
    # only reassigned at the TOP of the next call to forward_layer(), it
    # stays alive in GPU memory for the entire duration of computing the
    # *next* layer -- i.e. one full layer's worth of intermediates is held
    # redundantly, on top of whatever the next layer is actively computing.
    # Freeing it explicitly here shows exactly how much that is worth.
    del variables
    gc.collect()
    torch.cuda.empty_cache()
    log_mem(f"layer {layer_idx:02d} - after del(variables) + gc.collect() + empty_cache()")

    print("DONE")

total_bs_time = engine.bs_total_time()
total_keyload_time = timer.timers["key_timer"]
print(f"Key load time elapsed: {total_keyload_time}")
print(f"Bootstrapping time elapsed: {total_bs_time}")
print(f"Total bootstrap() calls: {_bootstrap_call_count[0]}")

if cache is not None:
    print(f"Cache policy: {CACHE_POLICY}")
    print(f"Cache hits: {cache._hits}")
    print(f"Cache misses: {cache._misses}")
    print(f"Cache hit ratio: {cache.hit_ratio:.4f}")
    print(f"Cache stats: {cache.cache_stats}")