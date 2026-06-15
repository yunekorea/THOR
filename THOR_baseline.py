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

    # ------------------------------------------------------------------
    # Core lookup – called as  bs_key[k]  by the bootstrapping internals
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        if key in self._gpu:
            # Cache hit → move to "most recently used" end
            self._gpu.move_to_end(key)
            return self._gpu[key]

        # Cache miss → load from CPU
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

params = {"logN":16, "scale_bits": 41, "num_special_primes": 4, "devices": devices, "quantum":"pre_quantum"}
engine = CkksEngine(params)
print("Memory allocated: ", torch.cuda.memory_allocated(devices[0]) /1024**3)


print("Key Loading: ", end="")
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

lru_cache = LRUBootstrapKeyCache(engine, host_store,
                                    max_gpu_keys=45)
engine.add_bs_key(lru_cache)
#engine.add_rot_keys_from_sk(deltas, sk)
print("DONE")
print("Memory allocated: ", torch.cuda.memory_allocated(devices[0]) /1024**3)


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

'''
print("Load and Run Plain Model:", end="")
model_plain  = thor.utils.load_model(dataset_type, f'./finetuned_models/{dataset_type}/model.safetensors')
model_plain.eval()
device = torch.device("cpu")
model_plain.to(device)
idx = 0
for batch in data_loader:
    print(idx, target_idx)
    if idx < target_idx:
        idx += 1
        continue
    elif idx == target_idx:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model_plain(**batch)
        break

def get_nonlinear_in_out(hidden_states, layer_idx):
    with torch.no_grad():
        bert_layer_m = model_plain.bert.encoder.layer[layer_idx] 
        attention_m = bert_layer_m.attention.self
        bert_output_m = model_plain.bert.encoder.layer[layer_idx].attention.output

        q = attention_m.transpose_for_scores(attention_m.query(hidden_states))
        k = attention_m.transpose_for_scores(attention_m.key(hidden_states))
        v = attention_m.transpose_for_scores(attention_m.value(hidden_states))
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(attention_m.attention_head_size)
        extended_att_mask = model_plain.get_extended_attention_mask(
                        attention_mask, 768
                    ).to(device)
        sfmtx_in = attention_scores+extended_att_mask
        att_probs_m = torch.nn.functional.softmax(sfmtx_in, dim=-1)
        sfmtx_out = att_probs_m
        att_context_m = torch.matmul(att_probs_m, v)
        context_layer = att_context_m.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attention_m.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        dense_output_m = bert_output_m.dense(context_layer)
        ln1_in = dense_output_m + hidden_states
        ln1_out = bert_output_m.LayerNorm(ln1_in)
        gelu_in = bert_layer_m.intermediate.dense(ln1_out)
        gelu_out = bert_layer_m.intermediate.intermediate_act_fn(gelu_in)
        dense2_out = bert_layer_m.output.dense(gelu_out)
        ln2_in = dense2_out + ln1_out
        ln2_out = bert_layer_m.output.LayerNorm(ln2_in)
        pooler_m = model_plain.bert.pooler
        pooler_dense_output = pooler_m.dense(ln2_out[:, 0])
        print(ln2_out[:, 0].shape)
        pooler_output = pooler_m.activation(pooler_dense_output)

    return (
        hidden_states.cpu().numpy().squeeze(),
        q.cpu().numpy().squeeze(),
        sfmtx_in.cpu().numpy().squeeze(),
        sfmtx_out.cpu().numpy().squeeze(),
        att_context_m.cpu().numpy().squeeze(),
        ln1_in.cpu().numpy().squeeze(),
        ln1_out.cpu().numpy().squeeze(),
        gelu_in.cpu().numpy().squeeze(),
        gelu_out.cpu().numpy().squeeze(),
        dense2_out.cpu().numpy().squeeze(),
        ln2_in.cpu().numpy().squeeze(),
        ln2_out.cpu().numpy().squeeze(),
        pooler_dense_output.cpu().numpy().squeeze(),
        pooler_output.cpu().numpy().squeeze()
        )
    
hidden_states = []
qs= []
ks = []
sftmx_ins = []
sftmx_outs = []
att_contexts = []
ln1_ins = []
ln1_outs = []
gelu_ins = []
gelu_outs = []
dense2_outs = []
ln2_ins = []
ln2_outs = []
for layer in range(12):
    hidden_state, q, sftmx_in, sftmx_out, att_context, ln1_in, ln1_out, gelu_in, gelu_out, dense2_out, ln2_in, ln2_out, pooler_dense_out, pooler_out = get_nonlinear_in_out(outputs.hidden_states[layer], layer)
    hidden_states.append(hidden_state)
    qs.append(q)
    sftmx_ins.append(sftmx_in)
    sftmx_outs.append(sftmx_out)
    att_contexts.append(att_context)
    ln1_ins.append(ln1_in)
    ln1_outs.append(ln1_out)
    gelu_ins.append(gelu_in)
    gelu_outs.append(gelu_out)
    dense2_outs.append(dense2_out)
    ln2_ins.append(ln2_in)
    ln2_outs.append(ln2_out)
print("DONE")
'''

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
    
    # WF 1. Attention layer
    x_cplx_rots = evaluator.make_rotated_copies(x_cplx)
    q_wo_rescale = thor_attention.query(x_cplx_rots)
    k = thor_attention.key(x_cplx_rots)
    v = thor_attention.value(x_cplx_rots)

    l_k = evaluator.transpose_upper_to_lower(k)
    l_k_cplx = np.full((4,), None, dtype=object)
    for i in range(4):
        l_k_cplx[i] = engine.cc_add(engine.level_up(l_k[i], l_k[i].level_calc+1), engine.imult(evaluator.rotate_internal(l_k[i], 64, mode='att')))
        l_k_cplx[i] = engine.rescale(l_k_cplx[i])
    
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
    
    # WF 3. Soft weights(Softmax)
    sftmx_out = thor_attention.softmax(x=sftmx_in, attention_mask=thor_attention_mask, rescale=False, debug=False, sk=None)

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
    # WF 4. Attention head & multi-head attention
    att_context = thor_attention.calculate_attention_context(v_cplx, sftmx_out_rescale, rescale=False)

    # Bootstrap #2
    for i in range(2):
        att_context[i] = engine.bootstrap(att_context[i])

    att_context_rots = thor_attention.evaluator.make_rotated_copies(att_context)
    dense_output = thor_attention.dense(att_context_rots)
    x_out_sum = np.full((8,), None, dtype=object)
    mask = np.array(([1]*6+[0]*10)*2**11)
    for i in range(4):
        x_out_sum[i] = engine.add(x[i], dense_output[i])
        x_out_sum[i+4] = engine.add(x[i+4], dense_output[i+4])
    ln1_in = x_out_sum

    # WF 5. LayerNorm1
    ln1_out = thor_attention.layernorm(x=ln1_in, sk=None)
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

    # WF 6. FC1(Fully Connected layer; Feed-Forward Network Part 1)
    gelu_in_wo_bs = thor_ff.dense1(l)

    for i in range(8):
        temp = engine.cc_add(gelu_in_wo_bs[0,i], engine.imult(gelu_in_wo_bs[1,i]))
        temp = engine.mult_scalar(temp, 1/2)
        #Bootstrap #3
        temp = engine.bootstrap(temp)
        conj = engine.conjugate(temp)
        gelu_in_wo_bs[0,i] = engine.cc_add(temp, conj)
        gelu_in_wo_bs[1,i] = engine.imult(engine.cc_sub(conj, temp))

    # WF 7. GELU(Activation Function)
    gelu_out = thor_ff.gelu(x=gelu_in_wo_bs)
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

    # WF 9. LayerNorm2
    if layer_idx == 9 or layer_idx == 10:
        ln2_out = thor_ff.layernorm(x=ln2_in, sk=None)
    else:
        ln2_out = thor_ff.layernorm(x=ln2_in, sk=None)

    if ln2_out[0].level >8:
        for i in range(8):
            ln2_out[i] = engine.level_up(ln2_out[i], 21)
        
    thor_attention.cpu()
    thor_ff.cpu()
    return ln2_out, (x, q_wo_rescale, sftmx_in, sftmx_out, att_context, ln1_in, ln1_out, gelu_in_wo_bs, gelu_out, dense2_out, ln2_in, ln2_out)

print("BASELINE test")

for layer_idx in range(12):
    print(f"Forwarding layer #{layer_idx}: ", end="")
    
    thor_attention = thor_bert.attentions[layer_idx]
    thor_ff = thor_bert.ffs[layer_idx]
    
    x, variables = forward_layer(x, layer_idx, thor_ff)  # ← update x each time
    
    print("DONE")
