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

def engine_init():
    params = {"logN":16, "scale_bits": 41, "num_special_primes": 4, "quantum":"pre_quantum"}
    engine = CkksEngine(params)
    return engine

def key_init(engine, key_path):
    rotk_dict_keys = [
        -32768, -16384, -1024, -512, -32, -16,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384,
        416, 448, 480, 512, 1024, 2048, 3072, 4096, 5120, 6144,
        7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336,
        15360, 16384
    ]
    pk = engine.load(f"{key_path}/pk")
    engine.add_pk(pk)
    evk = engine.load(f"{key_path}/evk")
    engine.add_evk(evk)
    conjk = engine.load(f"{key_path}/conjk")
    engine.add_conj_key(conjk)
    rotk_dict = {}
    for key in rotk_dict_keys:
        rotk_dict[key] = engine.load(f"{key_path}/rotk_dict/{key}")
    bs.create_cts_stc_const(engine)
    engine.add_bs_key(rotk_dict)

def main():
    #key_path = sys.argv[1]
    key_path = "/mnt/nvmf/THOR_test/THOR/keys/keys0"
    engine = engine_init()
    key_init(engine, key_path)

if __name__ == '__main__':
    main()