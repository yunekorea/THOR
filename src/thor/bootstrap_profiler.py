"""
bootstrap_profiler.py

CkksEngine.bootstrap() 안의 실제 연산 호출 한 줄
(`ct_bs = bs.bootstrap(self, temp, self.bs_key, self.evk, self.conj_key, self.pk)`,
ckks.py 98번째 줄)만 정밀하게 계측하기 위한 모듈.

설계 원칙
---------
- ckks.py는 THOR_baseline.py와 THOR_NDP_target.py(정확히는 offloading 받은 쪽인
  target)가 공통으로 거쳐가는 지점이라, 이 모듈을 ckks.py에서 딱 한 번만 연결하면
  두 시나리오 모두 자동으로 계측됩니다. 별도 프로파일링 함수를 두 곳에 따로
  작성할 필요가 없습니다.
- start()를 호출하지 않으면 track()은 사실상 아무 일도 하지 않는 콜(overhead
  거의 0)이라서, 프로파일링이 필요 없는 다른 스크립트가 ckks.py를 그냥 import해도
  안전합니다.
- 백그라운드 스레드가 짧은 주기(기본 0.05초)로 CPU / 호스트 DRAM / GPU
  util·VRAM / GPU<->호스트 PCIe 처리량 / 로컬 디스크 I/O / RDMA(NVMe-oF) I/O를
  계속 샘플링합니다. bootstrap 호출 하나하나를 mark()로 감싸서, 나중에
  "이 구간 동안 각 장치가 얼마나 바빴는지"를 정확히 잘라볼 수 있습니다.
- 지정한 일부 호출(기본: 처음 3번)에 한해서만 torch.profiler로 커널 단위까지
  파고듭니다. 레이어 전체를 감쌌던 예전 방식과 달리 bootstrap 호출 하나만
  감싸므로 이벤트 수가 훨씬 적어 chrome trace export도 안전합니다.
- finalize()에서 지금까지 모은 모든 mark 쌍(bs_N_start/bs_N_end)을 이용해
  bootstrap 호출별 소요시간 + 그 구간 동안의 평균 리소스 사용량을 CSV로
  요약해줍니다. baseline vs NDP target을 바로 비교할 수 있는 형태입니다.

사용법
------
ckks.py (이미 반영됨):
    from .bootstrap_profiler import bootstrap_profiler
    ...
    with bootstrap_profiler.track():
        ct_bs = bs.bootstrap(self, temp, self.bs_key, self.evk, self.conj_key, self.pk)

THOR_baseline.py:
    from thor.bootstrap_profiler import bootstrap_profiler
    bootstrap_profiler.start(out_dir="./profile_results/baseline",
                              gpu_index=devices[0], ib_device="enp216s0np0")
    ... (평소처럼 실행) ...
    bootstrap_profiler.finalize()   # 모든 forwarding이 끝난 뒤

THOR_NDP_target.py:
    from thor.bootstrap_profiler import bootstrap_profiler
    bootstrap_profiler.start(out_dir="./profile_results/ndp_target",
                              gpu_index=devices[0], ib_device="enp216s0np0")
    ... (셀렉터 이벤트 루프) ...
    except KeyboardInterrupt:
        bootstrap_profiler.finalize()   # 종료 명령이 들어온 시점
"""

import os
import re
import csv
import json
import time
import threading
import subprocess
from contextlib import contextmanager

import psutil

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False


class BootstrapProfiler:
    def __init__(self):
        self.enabled = False
        self._lock = threading.Lock()

    # ---------------- 시작 ----------------
    def start(self, out_dir=".", gpu_index=0, sample_interval=0.05,
              ib_device=None, ib_port=1,
              detailed_profile_calls=None, detailed_profile_every=None,
              max_detailed_calls=5, checkpoint_every_calls=50,
              nvidia_smi_min_interval=1.0):
        """
        detailed_profile_calls : 명시적으로 torch.profiler를 걸 호출 번호 집합 (1-based).
                                  예: {1, 2, 3, 100, 200}
        detailed_profile_every : N번째 호출마다 torch.profiler (detailed_profile_calls보다 낮은 우선순위).
        max_detailed_calls      : 위 두 옵션으로 정밀 계측할 총 횟수 상한 (오버헤드 보호).
        checkpoint_every_calls  : 이 호출 수마다 중간 저장(체크포인트) 수행.
                                  NDP target처럼 오래 켜져 있는 프로세스가 중간에 죽어도
                                  데이터를 잃지 않기 위함.
        """
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self.detailed_profile_calls = set(detailed_profile_calls or [])
        self.detailed_profile_every = detailed_profile_every
        self.max_detailed_calls = max_detailed_calls
        self.checkpoint_every_calls = checkpoint_every_calls
        self._detailed_done = 0

        self.process = psutil.Process(os.getpid())
        self.gpu_available = _HAS_TORCH and torch.cuda.is_available()
        self._nvml_handle = None
        if self.gpu_available and _HAS_PYNVML:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:
                print(f"[BootstrapProfiler] pynvml handle failed ({e}); "
                      f"falling back to nvidia-smi subprocess polling.")

        self.nvidia_smi_min_interval = nvidia_smi_min_interval
        self._last_nvsmi_poll_t = -1.0
        self._last_nvsmi_result = (None, None, None)

        # RDMA(InfiniBand/RoCE) 카운터. 커널 문서 기준 4바이트(레인) 단위로 노출되므로 x4.
        self.ib_device = ib_device
        self.ib_port = ib_port
        self._ib_rx_path = None
        self._ib_tx_path = None
        self.rdma_available = False
        if ib_device:
            rx_path = f"/sys/class/infiniband/{ib_device}/ports/{ib_port}/counters/port_rcv_data"
            tx_path = f"/sys/class/infiniband/{ib_device}/ports/{ib_port}/counters/port_xmit_data"
            try:
                with open(rx_path) as f:
                    int(f.read().strip())
                with open(tx_path) as f:
                    int(f.read().strip())
                self._ib_rx_path, self._ib_tx_path = rx_path, tx_path
                self.rdma_available = True
            except Exception as e:
                print(f"[BootstrapProfiler] RDMA counters unavailable ({e}); RDMA tracking disabled.")

        self.samples = []
        self.marks = []
        self.detailed_records = []  # 정밀 프로파일링된 호출들의 메타 정보

        self._call_count = 0
        self._stop_flag = threading.Event()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self.enabled = True

        gpu_msg = "no GPU" if not self.gpu_available else (
            "pynvml" if self._nvml_handle is not None else "nvidia-smi fallback")
        rdma_msg = f"RDMA={ib_device}" if self.rdma_available else "RDMA=off"
        print(f"[BootstrapProfiler] started. out_dir={out_dir} gpu=({gpu_msg}) {rdma_msg}")

    # ---------------- 백그라운드 샘플링 ----------------
    def _read_ib_counters(self):
        with open(self._ib_rx_path) as f:
            rx = int(f.read().strip())
        with open(self._ib_tx_path) as f:
            tx = int(f.read().strip())
        return rx * 4, tx * 4

    def _gpu_stats_pynvml(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        pcie_tx = pcie_rx = None
        try:
            pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(self._nvml_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(self._nvml_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
        except Exception:
            pass
        return float(util.gpu), mem.used / (1024 ** 2), mem.total / (1024 ** 2), pcie_tx, pcie_rx

    def _gpu_stats_nvidia_smi(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0,
            )
            util, used, total = [float(v) for v in out.stdout.strip().split(",")]
            return util, used, total
        except Exception:
            return None, None, None

    def _sample_loop(self):
        self.process.cpu_percent(interval=None)
        last_io = None
        try:
            last_io = self.process.io_counters()
        except (psutil.AccessDenied, NotImplementedError, AttributeError):
            pass

        last_ib = None
        if self.rdma_available:
            try:
                last_ib = self._read_ib_counters()
            except Exception:
                self.rdma_available = False

        while not self._stop_flag.is_set():
            now = time.perf_counter() - self._t0
            vm = psutil.virtual_memory()
            sample = {
                "t": round(now, 4),
                "cpu_percent": self.process.cpu_percent(interval=None),
                "rss_mb": round(self.process.memory_info().rss / (1024 ** 2), 2),
                "dram_used_mb": round(vm.used / (1024 ** 2), 2),
                "dram_percent": vm.percent,
            }

            if last_io is not None:
                try:
                    io = self.process.io_counters()
                    dt = self.sample_interval
                    sample["disk_read_mb_s"] = round((io.read_bytes - last_io.read_bytes) / (1024 ** 2) / dt, 3)
                    sample["disk_write_mb_s"] = round((io.write_bytes - last_io.write_bytes) / (1024 ** 2) / dt, 3)
                    last_io = io
                except Exception:
                    pass

            if self.gpu_available:
                sample["gpu_alloc_mb"] = round(torch.cuda.memory_allocated(self.gpu_index) / (1024 ** 2), 2)
                sample["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(self.gpu_index) / (1024 ** 2), 2)

                if self._nvml_handle is not None:
                    util, used, total, pcie_tx, pcie_rx = self._gpu_stats_pynvml()
                    if pcie_tx is not None:
                        sample["gpu_pcie_tx_kb_s"] = pcie_tx  # nvml: KB/s, ~20ms 윈도우 평균
                        sample["gpu_pcie_rx_kb_s"] = pcie_rx
                elif now - self._last_nvsmi_poll_t >= self.nvidia_smi_min_interval:
                    util, used, total = self._gpu_stats_nvidia_smi()
                    self._last_nvsmi_poll_t = now
                    self._last_nvsmi_result = (util, used, total)
                else:
                    util, used, total = self._last_nvsmi_result
                if util is not None:
                    sample["gpu_util_percent"] = util
                    sample["gpu_mem_used_mb"] = used
                    sample["gpu_mem_total_mb"] = total

            if self.rdma_available and last_ib is not None:
                try:
                    rx, tx = self._read_ib_counters()
                    dt = self.sample_interval
                    sample["rdma_rx_mb_s"] = round((rx - last_ib[0]) / (1024 ** 2) / dt, 3)
                    sample["rdma_tx_mb_s"] = round((tx - last_ib[1]) / (1024 ** 2) / dt, 3)
                    last_ib = (rx, tx)
                except Exception:
                    pass

            self.samples.append(sample)
            self._stop_flag.wait(self.sample_interval)

    # ---------------- 마커 ----------------
    def mark(self, stage):
        now = time.perf_counter() - self._t0
        self.marks.append({"t": round(now, 4), "stage": stage})

    def _should_profile_detailed(self, n):
        if self._detailed_done >= self.max_detailed_calls:
            return False
        if n in self.detailed_profile_calls:
            return True
        if self.detailed_profile_every and n % self.detailed_profile_every == 0:
            return True
        return False

    # ---------------- 호출 하나를 감싸는 컨텍스트 매니저 ----------------
    @contextmanager
    def track(self):
        if not self.enabled:
            yield
            return

        with self._lock:
            self._call_count += 1
            n = self._call_count
        do_detailed = self._should_profile_detailed(n)

        self.mark(f"bs_{n:05d}_start")

        if do_detailed and _HAS_TORCH and self.gpu_available:
            from torch.profiler import profile, ProfilerActivity
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=True, with_stack=False,
            ) as tp:
                yield
            self._save_detailed(n, tp)
            self._detailed_done += 1
        else:
            yield

        self.mark(f"bs_{n:05d}_end")

        if self.checkpoint_every_calls and n % self.checkpoint_every_calls == 0:
            self._checkpoint_save()

    def _save_detailed(self, n, tp):
        key_avgs = tp.key_averages()
        n_events = len(key_avgs)
        table = key_avgs.table(sort_by="self_cuda_time_total", row_limit=40)
        txt_path = os.path.join(self.out_dir, f"bootstrap_call_{n:05d}_torch_profile.txt")
        with open(txt_path, "w") as f:
            f.write(table)
        print(f"[BootstrapProfiler] call #{n}: detailed profile saved -> {txt_path} "
              f"({n_events} unique ops)")

        # 단일 bootstrap 호출은 이벤트 수가 훨씬 적어서(레이어 전체 대비 수십분의 1)
        # chrome trace export가 보통 안전합니다. 그래도 혹시 모를 폭주에 대비해
        # 이벤트 수가 과하면 건너뜁니다.
        if n_events < 20000:
            try:
                trace_path = os.path.join(self.out_dir, f"bootstrap_call_{n:05d}_trace.json")
                tp.export_chrome_trace(trace_path)
                print(f"[BootstrapProfiler] call #{n}: chrome trace saved -> {trace_path}")
            except Exception as e:
                print(f"[BootstrapProfiler] call #{n}: chrome trace export failed ({e}); skipped.")
        else:
            print(f"[BootstrapProfiler] call #{n}: {n_events} unique ops is a lot -- "
                  f"skipping chrome trace export to avoid a multi-minute stall.")

        self.detailed_records.append({"call": n, "n_events": n_events, "txt_path": txt_path})

    # ---------------- 체크포인트 / 종료 ----------------
    def _checkpoint_save(self):
        try:
            path = os.path.join(self.out_dir, "bootstrap_profile_checkpoint.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"samples": self.samples, "marks": self.marks}, f)
        except Exception as e:
            print(f"[BootstrapProfiler] checkpoint save failed ({e})")

    def finalize(self):
        if not self.enabled:
            return
        self.enabled = False  # 이후 track() 호출은 즉시 no-op으로 빠짐
        self._stop_flag.set()
        self._thread.join(timeout=5.0)

        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "bootstrap_profile_samples.json"), "w", encoding="utf-8") as f:
            json.dump({"samples": self.samples, "marks": self.marks}, f, indent=2)

        if self.samples:
            keys = sorted({k for s in self.samples for k in s.keys()})
            with open(os.path.join(self.out_dir, "bootstrap_profile_samples.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.samples)

        self._write_call_summary()
        print(f"[BootstrapProfiler] finalized. total bootstrap calls tracked: {self._call_count}. "
              f"results in {self.out_dir}")

    def _write_call_summary(self):
        """bs_N_start/bs_N_end 마커 쌍마다 소요시간 + 그 구간 평균 리소스 사용량을 CSV로 저장."""
        starts, ends = {}, {}
        for m in self.marks:
            mm = re.match(r"bs_(\d+)_(start|end)", m["stage"])
            if not mm:
                continue
            call_id, kind = int(mm.group(1)), mm.group(2)
            (starts if kind == "start" else ends)[call_id] = m["t"]

        numeric_keys = sorted({k for s in self.samples for k, v in s.items()
                                if k != "t" and isinstance(v, (int, float))})
        rows = []
        for call_id in sorted(starts):
            if call_id not in ends:
                continue  # 진행 중이던 마지막 호출 (셧다운 시 중간에 끊긴 경우)
            t0, t1 = starts[call_id], ends[call_id]
            in_range = [s for s in self.samples if t0 <= s["t"] <= t1]
            row = {"call_id": call_id, "start_t": t0, "end_t": t1, "duration_s": round(t1 - t0, 4),
                   "n_samples": len(in_range)}
            for k in numeric_keys:
                vals = [s[k] for s in in_range if k in s]
                row[f"mean_{k}"] = round(sum(vals) / len(vals), 4) if vals else None
            rows.append(row)

        if not rows:
            return
        path = os.path.join(self.out_dir, "bootstrap_call_summary.csv")
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        durations = [r["duration_s"] for r in rows]
        durations_sorted = sorted(durations)
        n = len(durations_sorted)
        median = durations_sorted[n // 2] if n % 2 else (durations_sorted[n // 2 - 1] + durations_sorted[n // 2]) / 2
        print(f"[BootstrapProfiler] {n} completed bootstrap calls | "
              f"duration mean={sum(durations)/n:.3f}s median={median:.3f}s "
              f"min={min(durations):.3f}s max={max(durations):.3f}s")
        print(f"[BootstrapProfiler] per-call summary -> {path}")


# 모듈 전역 싱글턴. ckks.py, THOR_baseline.py, THOR_NDP_target.py가 전부 이걸 공유합니다.
bootstrap_profiler = BootstrapProfiler()
