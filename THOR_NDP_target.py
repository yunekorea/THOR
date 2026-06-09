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

import time

sel = selectors.DefaultSelector()

def engine_init():
    print("engine init: ", end="")
    params = {"logN":16, "scale_bits": 41, "num_special_primes": 4, "quantum":"pre_quantum"}
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
    rotk_dict = {}
    numkeys = len(rotk_dict_keys)
    loaded_stat = 0
    for key in rotk_dict_keys:
        loaded_dict_key = engine.load(f"{key_path}/rotk_dict/{key}")
        rotk_dict[key] = loaded_dict_key
        del loaded_dict_key
        gc.collect()
        torch.cuda.empty_cache()
        loaded_stat += 1
        print(f"loaded keys: {loaded_stat}/{numkeys}")
    bs.create_cts_stc_const(engine)
    engine.add_bs_key(rotk_dict)

    del rotk_dict
    gc.collect()
    print("DONE")
    
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
    print("ROTK dict: ", end="")

def RDMA_init():
    dev_name = "mlx5_0".encode('utf-8')
    dev_name_len = len(dev_name)
    ctx = Context(name='mlx5_0')
    
    cap = QPCap(max_send_wr=16, max_recv_wr=16, max_send_sge=8)
    qp_init_attr = QPInitAttr(cap=cap)

    # Initialize CIMD
    host_ip = "192.168.100.2"
    target_ip = "192.168.100.1"
    cai = AddrInfo(src = target_ip,dst=host_ip, dst_service="9999",
                    port_space = rdma_port_space.RDMA_PS_TCP)
    cid = CMID(creator=cai, qp_init_attr=qp_init_attr)
    return cid

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

def read_ciphertext(conn, mask, cid):
    data = conn.recv(128)
    if data:
        print("Interrupt received! Processing metadata...")
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
            
            print(f"--- Decoded Metadata ---")
            print(f"Address:     {hex(addr)}")
            print(f"R-Key:       {hex(rkey)}")
            print(f"Length:      {length}")
            print(f"Name Length: {name_length}")
            print(f"Device Name: {device_name}")
            
            # Now you can proceed with your RDMA logic using these variables
            print("cid.connect() start")
            cid.connect()
        
            local_mr = cid.reg_msgs(length)
            print("MR set")
            
            if 1: # Success
                print("RDMA Read Successful!")
                # Verify by reading the local buffer content
                print(f"Data from Host: {local_mr.read(32, 0).decode()}")
            else:
                #print(f"RDMA Read Failed. Status code: {wc.status}")
                print("ff")

            print("Post SEND")
            cid.post_send(local_mr, length)
            print("Successful")


        
        except struct.error as e:
            print(f"Error unpacking metadata: {e}")
        except Exception as e:
            print(f"RDMA Operation error: {e}")

    sel.unregister(conn)
    conn.close()

def accept_connection(sock, mask, cid):
    conn, _ = sock.accept()
    conn.setblocking(False)
    # Register the 'read' event for this specific connection
    sel.register(conn, selectors.EVENT_READ,
                data=lambda key_obj, mask_val: read_ciphertext(key_obj.fileobj, mask_val, cid))

def main():
    #key_path = sys.argv[1]
    key_path = "/mnt/nvmf/THOR_test/THOR/keys/keys0"
    engine = engine_init()
    key_init(engine, key_path)
    cid = RDMA_init()
    server = UDS_init()
    sel.register(server, selectors.EVENT_READ,
                 data=lambda key_obj, mask_val: accept_connection(key_obj.fileobj, mask_val, cid))

    print("Python is ready. Waiting for asynchronous events...")
    try:
        while True:
            events = sel.select() # This blocks efficiently (uses epoll/kqueue)
            for key, mask in events:
                callback = key.data
                callback(key.fileobj, mask)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == '__main__':
    main()
