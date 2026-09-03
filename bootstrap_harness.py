"""
bootstrap_harness.py

목적
----
"NDP가 빠른 이유가 GPU 하드웨어 때문인가, 아니면 프로세스가 가벼워서인가"를
분리해서 측정하기 위한 최소 하네스입니다.

지금까지의 측정은 GPU와 프로세스 부하가 항상 함께 움직여서(A6000=풀 워크로드,
RTX3090=bootstrap 전용) 두 변수를 분리할 수 없었습니다. 이 스크립트는
bootstrap만 수행하는 최소 프로세스를 만들어, 아래 2x2를 채울 수 있게 합니다.

                     |  A6000(Host)        |  RTX3090(Target)
    -----------------+---------------------+--------------------
    최소 하네스      |  (a) <- 이 스크립트  |  (b) <- 이 스크립트
    풀 워크로드      |  (c) 기존 baseline   |  (d) 기존 NDP target

  * (a) vs (c) : GPU 동일, 프로세스 부하만 다름 -> 오프로딩의 순수 효과
  * (a) vs (b) : 프로세스 동일, GPU만 다름     -> 하드웨어의 순수 효과

측정 방법은 기존과 동일하게 bootstrap_profiler 를 사용하므로, 출력되는
bootstrap_call_summary.csv 를 기존 결과와 그대로 나란히 비교할 수 있습니다.

전제
----
- THOR_baseline.py 와 같은 디렉토리에서 실행 (keys/ 경로가 상대경로라서).
- thor 패키지가 import 가능해야 함 (기존 스크립트와 동일 환경).
- 캐시 클래스는 THOR_baseline.py 에서 그대로 복사해 왔습니다. 원본이 바뀌면
  이 파일도 같이 갱신해야 합니다.

실행 예시
--------
  P=/home/yunekorea/.venv/310THOR/bin/python

  # (a) A6000 에서 최소 하네스 - 가장 중요한 새 데이터
  sudo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P bootstrap_harness.py \
       --iters 60 --out ./profile_results/harness_a6000

  # (b) RTX3090(Target) 에서 최소 하네스
  sudo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P bootstrap_harness.py \
       --iters 60 --out ./profile_results/harness_3090

  # 참고: A6000 에서 "풀 워크로드와 비슷한 VRAM 점유"를 흉내내고 싶으면
  sudo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P bootstrap_harness.py \
       --iters 60 --ballast-gb 12 --out ./profile_results/harness_a6000_ballast

level 인자에 대하여
------------------
bootstrap 은 특정 level 의 암호문을 입력으로 받습니다. 실제 THOR 에서 어떤
level 로 들어오는지 확실치 않으면 --probe-level 로 먼저 탐색하세요.
탐색 모드는 여러 level 을 시도해서 성공 여부와 1회 소요시간을 표로 보여줍니다.
실제 값과 다른 level 로 측정하면 연산량이 달라져 비교가 무의미해지므로,
반드시 실제 THOR 실행의 bootstrap 입력 level 과 맞추세요.
(가장 확실한 방법: ckks.py 의 bootstrap() 안에 print(ct.level, ct.level_calc)
 를 한 줄 넣고 THOR_baseline.py 를 잠깐 돌려 첫 값을 확인)
"""

import sys
import os
from collections import OrderedDict
project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)
project_root = os.path.abspath(os.path.join(os.getcwd(), '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import gc
import statistics as st
import time

import numpy as np
import torch

import thor
from thor import CkksEngine
from thor.bootstrap_profiler import bootstrap_profiler
from liberate.fhe.bootstrapping import ckks_bootstrapping as bs

# THOR_baseline.py 와 동일한 rotation key 목록 (55개)
rotk_dict_keys = [
    -32768, -16384, -1024, -512, -32, -16,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384,
    416, 448, 480, 512, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336,
    15360, 16384
]


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




# ----------------------------------------------------------------------
# 하네스 본체
# ----------------------------------------------------------------------

def build_engine(devices, key_path, load_gk=False, settle_sleep=10):
    """엔진 + 키 준비.

    중요: 로딩 순서와 종류를 THOR_NDP_target.py 의 key_init() 과 동일하게 맞췄습니다.
      - gk(Galois key)는 로드하지 않습니다. Target 은 bootstrap 만 하므로 필요 없고,
        gk 는 rotation key 급으로 커서 24GB GPU 에서는 이것만으로 OOM 이 납니다.
        (Host 의 baseline 은 forward_layer 에서 rotate 를 쓰기 때문에 gk 를 올립니다.)
      - rotation key -> 캐시 부착 -> pk/evk/conjk 순서입니다. baseline 과 반대인데,
        Target 은 이 순서라야 24GB 에 들어갑니다.
      - 각 키는 engine.add_*() 로 넘긴 직후 del + gc.collect() 로 중복 참조를 끊고,
        evk 뒤에는 empty_cache() + 짧은 대기로 반환을 확실히 합니다.
    load_gk=True 로 주면 baseline 과 동일하게 gk 까지 올립니다(48GB GPU 전용).
    """
    params = {"logN": 16, "scale_bits": 41, "num_special_primes": 4,
              "devices": devices, "quantum": "pre_quantum"}
    engine = CkksEngine(params)
    print(f"[harness] engine ready. alloc={torch.cuda.memory_allocated(devices[0])/1024**3:.3f}GB")

    gc.collect()
    torch.cuda.empty_cache()

    host_store = {}
    for key in rotk_dict_keys:
        host_store[key] = engine.load(f"{key_path}/rotk_dict/{key}", move_to_gpu=False)
    print(f"[harness] {len(host_store)} rotation keys loaded to host DRAM")

    bs.create_cts_stc_const(engine)
    print(f"[harness] after create_cts_stc_const: "
          f"alloc={torch.cuda.memory_allocated(devices[0])/1024**3:.3f}GB")
    return engine, host_store


def load_remaining_keys(engine, key_path, gpu, load_gk=False, settle_sleep=10):
    """캐시 부착 후에 pk/evk/(gk)/conjk 를 올립니다. Target 과 동일한 순서/정리 방식."""
    print("[harness] pk: ", end="", flush=True)
    pk = engine.load(f"{key_path}/pk")
    engine.add_pk(pk)
    del pk
    gc.collect()
    print("DONE")

    print("[harness] evk: ", end="", flush=True)
    evk = engine.load(f"{key_path}/evk")
    engine.add_evk(evk)
    del evk
    gc.collect()
    print("DONE")
    torch.cuda.empty_cache()
    if settle_sleep:
        time.sleep(settle_sleep)

    if load_gk:
        print("[harness] gk: ", end="", flush=True)
        gk = engine.load(f"{key_path}/gk")
        engine.add_gk(gk)
        del gk
        gc.collect()
        torch.cuda.empty_cache()
        print("DONE")

    print("[harness] conjk: ", end="", flush=True)
    conjk = engine.load(f"{key_path}/conjk")
    engine.add_conj_key(conjk)
    del conjk
    gc.collect()
    print("DONE")
    print(f"[harness] after pk/evk/conjk: "
          f"alloc={torch.cuda.memory_allocated(gpu)/1024**3:.3f}GB "
          f"reserved={torch.cuda.memory_reserved(gpu)/1024**3:.3f}GB")


def attach_cache(engine, host_store, policy, max_gpu_keys):
    if policy == "lru":
        cache = LRUBootstrapKeyCache(engine, host_store, max_gpu_keys=max_gpu_keys)
        engine.add_bs_key(cache)
    elif policy == "belady":
        cache = BeladyBootstrapKeyCache(engine, host_store, max_gpu_keys=max_gpu_keys)
        engine.add_bs_key(cache)
    elif policy == "none":
        gpu_key_dict = {k: engine.cuda(v) for k, v in host_store.items()}
        engine.add_bs_key(gpu_key_dict)
        cache = None
    else:
        raise ValueError(f"Unknown cache policy: {policy!r}")
    return cache


def make_ct(engine, level):
    """bootstrap 입력용 더미 암호문 생성. pk 는 엔진에 등록된 것을 사용."""
    n_slots = engine.num_slots if hasattr(engine, "num_slots") else (1 << 15)
    m = np.random.randn(n_slots) * 0.01
    return engine.encode_and_encrypt(m, level=level)


def probe_levels(engine, lo, hi):
    print(f"\n[harness] level 탐색: {lo} ~ {hi}")
    print(f"  {'level':>6s} {'결과':>10s} {'소요시간':>12s}")
    ok = []
    for lv in range(lo, hi + 1):
        try:
            ct = make_ct(engine, lv)
        except Exception as e:
            print(f"  {lv:6d} {'암호화실패':>10s}   {type(e).__name__}: {str(e)[:40]}")
            continue
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = engine.bootstrap(ct)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"  {lv:6d} {'성공':>10s} {dt:11.3f}s")
            ok.append((lv, dt))
            del out
        except Exception as e:
            print(f"  {lv:6d} {'bs실패':>10s}   {type(e).__name__}: {str(e)[:40]}")
        finally:
            del ct
            gc.collect()
            torch.cuda.empty_cache()
    if ok:
        print(f"\n  성공한 level: {[l for l, _ in ok]}")
        print("  실제 THOR 실행의 bootstrap 입력 level 과 일치하는 값을 --level 로 지정하세요.")
    else:
        print("\n  성공한 level 이 없습니다. 키 경로나 엔진 파라미터를 확인하세요.")
    return ok


def make_ballast(dev, gb):
    """풀 워크로드의 VRAM 점유를 흉내내기 위한 상주 블록 (306MB 대형 + 17MB 중형)."""
    if gb <= 0:
        return []
    target = int(gb * 1024**3)
    blocks, got = [], 0
    big = 306 * 1024 * 1024
    small = 17 * 1024 * 1024
    while got < int(target * 0.7):
        try:
            blocks.append(torch.empty(big, dtype=torch.uint8, device=dev))
        except torch.cuda.OutOfMemoryError:
            break
        got += big
    while got < target:
        try:
            blocks.append(torch.empty(small, dtype=torch.uint8, device=dev))
        except torch.cuda.OutOfMemoryError:
            break
        got += small
    print(f"[harness] ballast {got/1024**3:.2f}GB ({len(blocks)} blocks)")
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--key-path", default="/mnt/nvmf/THOR_test/THOR/keys/keys0")
    ap.add_argument("--cache-policy", default="belady", choices=["belady", "lru", "none"])
    ap.add_argument("--max-gpu-keys", type=int, default=49)
    ap.add_argument("--level", type=int, default=None,
                    help="bootstrap 입력 암호문의 level. 모르면 --probe-level 먼저.")
    ap.add_argument("--probe-level", action="store_true")
    ap.add_argument("--probe-range", default="0,24")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--ballast-gb", type=float, default=0.0)
    ap.add_argument("--load-gk", action="store_true",
                    help="Galois key 까지 로드 (baseline 과 동일). 24GB GPU 에서는 OOM 위험.")
    ap.add_argument("--settle-sleep", type=float, default=10,
                    help="evk 로드 후 메모리 반환 대기(초). Target 원본과 동일하게 기본 10.")
    ap.add_argument("--out", default="./profile_results/harness")
    ap.add_argument("--ib-device", default=None, help="예: rocep59s0 (Host) / mlx5_0 (Target)")
    ap.add_argument("--detailed", default="1,2,3,40,50",
                    help="torch.profiler 로 상세 계측할 호출 번호")
    args = ap.parse_args()

    devices = [args.gpu]
    torch.cuda.set_device(args.gpu)
    props = torch.cuda.get_device_properties(args.gpu)
    print("=" * 70)
    print(f"[harness] GPU: {props.name}  VRAM {props.total_memory/1024**3:.1f}GB")
    print(f"[harness] PYTORCH_CUDA_ALLOC_CONF = "
          f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(기본)')}")
    print(f"[harness] cache={args.cache_policy} max_gpu_keys={args.max_gpu_keys}")
    print("=" * 70)

    engine, host_store = build_engine(devices, args.key_path)

    if args.probe_level:
        lo, hi = [int(x) for x in args.probe_range.split(",")]
        attach_cache(engine, host_store, args.cache_policy, args.max_gpu_keys)
        load_remaining_keys(engine, args.key_path, args.gpu,
                            load_gk=args.load_gk, settle_sleep=args.settle_sleep)
        probe_levels(engine, lo, hi)
        return

    if args.level is None:
        raise SystemExit("--level 을 지정하거나 --probe-level 로 먼저 탐색하세요.")

    ballast = make_ballast(torch.device(f"cuda:{args.gpu}"), args.ballast_gb)
    # Target 과 동일한 순서: rotation key 캐시를 먼저 붙이고 그 다음 pk/evk/conjk
    cache = attach_cache(engine, host_store, args.cache_policy, args.max_gpu_keys)
    load_remaining_keys(engine, args.key_path, args.gpu,
                        load_gk=args.load_gk, settle_sleep=args.settle_sleep)

    detailed = {int(x) for x in args.detailed.split(",") if x.strip()}
    bootstrap_profiler.start(out_dir=args.out, gpu_index=args.gpu,
                             ib_device=args.ib_device,
                             detailed_profile_calls=detailed)

    ct = make_ct(engine, args.level)
    times = []
    for i in range(1, args.iters + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = engine.bootstrap(ct)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        # 실제 THOR 와 동일하게 매 bootstrap 후 사이클 종료를 알림
        if cache is not None:
            cache.end_of_cycle()
        print(f"[harness] bootstrap {i:4d}/{args.iters}  {dt:7.3f}s  "
              f"alloc={torch.cuda.memory_allocated(args.gpu)/1024**3:6.3f}GB")
        del out

    bootstrap_profiler.finalize()

    warm = times[3:] if len(times) > 3 else times
    print("\n" + "=" * 70)
    print(f"[harness] {len(times)}회 완료 (앞 3회 제외한 {len(warm)}회 기준)")
    print(f"  mean   = {st.mean(warm):.4f}s")
    print(f"  median = {st.median(warm):.4f}s")
    if len(warm) > 1:
        print(f"  stdev  = {st.stdev(warm):.4f}s   (변동계수 "
              f"{st.stdev(warm)/st.mean(warm)*100:.2f}%)")
    print(f"  min/max= {min(warm):.4f}s / {max(warm):.4f}s")
    if cache is not None:
        print(f"  cache  = {cache.cache_stats}")
    print(f"  결과 CSV: {args.out}/bootstrap_call_summary.csv")
    print("=" * 70)

    del ballast
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
