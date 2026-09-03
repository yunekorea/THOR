import selectors
import socket
import sys
import os
import pprint
import mmap
import struct

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
from unittest.mock import patch

import time

sel = selectors.DefaultSelector()

print("engine init: ", end="")
params = {"logN":16, "scale_bits": 41, "num_special_primes": 4, "quantum":"pre_quantum"}
engine = CkksEngine(params)
print("DONE")

key_path = "/mnt/nvmf/THOR_test/THOR/keys/keys0"

print("key init:")
rotk_dict_keys = [
    -32768, -16384, -1024, -512, -32, -16,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384,
    416, 448, 480, 512, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336,
    15360, 16384
]

rotk_dict = {}
numkeys = len(rotk_dict_keys)
loaded_stat = 0
for key in rotk_dict_keys:
    loaded_dict_key = engine.load(f"{key_path}/rotk_dict/{key}", move_to_gpu = False)
    rotk_dict[key] = loaded_dict_key
    del loaded_dict_key
    loaded_stat += 1
    print(f"loaded keys: {loaded_stat}/{numkeys}")

bs.create_cts_stc_const(engine)
engine.add_bs_key(rotk_dict)

gc.collect()
torch.cuda.empty_cache()

print("DONE")
