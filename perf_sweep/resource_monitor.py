"""资源监控：CPU / 系统内存 / NPU AI core / NPU 显存。

后台线程定期采样；run_one 期间 start() → 跑测试 → stop() → 拿聚合结果合并到 CSV 行。

数据源：
  - CPU%   : /proc/stat 两次采样差分（不依赖 psutil）
  - 内存%  : /proc/meminfo (MemTotal - MemAvailable) / MemTotal
  - NPU    : npu-smi info 解析，正则匹配每个 chip 的 "AICore(%)  Memory-Usage(MB)" 行

非 Ascend 机器（如本机调试）npu-smi 不存在时优雅降级，NPU 列留空。
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# npu-smi subprocess 单次超时。stop() 的 join 超时基于这个值，保证 join
# 不会在 worker 仍卡在 npu-smi 里就返回。
_NPU_SMI_TIMEOUT_S = 5.0
MIN_INTERVAL_S = 0.5  # 公开常量；orchestrator 用它把"展示给用户的间隔"对齐到实际间隔


def effective_interval(requested_s: float) -> float:
    """ResourceMonitor 内部 clamp 的同步函数，便于 orchestrator 提前显示真实间隔。"""
    return max(MIN_INTERVAL_S, float(requested_s))


# ---------------------- CPU / 内存（/proc） ----------------------
def _read_cpu_jiffies() -> tuple[int, int] | None:
    """返回 (idle_jiffies, total_jiffies)。"""
    try:
        with open("/proc/stat", "r") as f:
            first = f.readline()
    except (OSError, FileNotFoundError):
        return None
    if not first.startswith("cpu "):
        return None
    parts = first.split()[1:]
    try:
        nums = [int(x) for x in parts]
    except ValueError:
        return None
    if len(nums) < 5:
        return None
    idle = nums[3] + nums[4]  # idle + iowait
    total = sum(nums)
    return idle, total


def _cpu_percent(prev: tuple[int, int] | None) -> tuple[float | None, tuple[int, int] | None]:
    cur = _read_cpu_jiffies()
    if cur is None or prev is None:
        return None, cur
    di = cur[0] - prev[0]
    dt = cur[1] - prev[1]
    if dt <= 0:
        return None, cur
    pct = 100.0 * (1.0 - di / dt)
    return max(0.0, min(100.0, pct)), cur


def _mem_percent() -> float | None:
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                k, _, rest = line.partition(":")
                tok = rest.strip().split()
                if tok and tok[0].isdigit():
                    info[k] = int(tok[0])  # kB
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if not total or avail is None:
            return None
        return 100.0 * (total - avail) / total
    except (OSError, FileNotFoundError):
        return None


# ---------------------- NPU（npu-smi） ----------------------
# npu-smi info 输出里每个 chip 占 2 行，关键的第二行形如：
#   | 0                | 0000:81:00.0   | 0           1234 / 65536        |
# 容错：AICore 可能是整数或浮点；Memory-Usage 单位 MB；分隔符可能空格也可能 /
_NPU_LINE_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*([\d.]+)\s+(\d+)\s*/\s*(\d+)"
)


def _read_npu_smi() -> list[tuple[int, float, int, int]]:
    """返回每个 chip 的 (chip_id, aicore_pct, mem_used_MB, mem_total_MB)。
    npu-smi 不可用或解析失败返回空列表。"""
    try:
        proc = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True,
            timeout=_NPU_SMI_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    rows: list[tuple[int, float, int, int]] = []
    for line in proc.stdout.splitlines():
        m = _NPU_LINE_RE.search(line)
        if m:
            try:
                rows.append((int(m.group(1)), float(m.group(2)),
                             int(m.group(3)), int(m.group(4))))
            except ValueError:
                continue
    return rows


# ---------------------- 监控线程 ----------------------
EMPTY_RESULT = {
    "cpu_util_avg(%)": "", "cpu_util_max(%)": "",
    "mem_util_avg(%)": "", "mem_util_max(%)": "",
    "npu_aicore_avg(%)": "", "npu_aicore_max(%)": "",
    "npu_mem_used_max(MB)": "", "npu_mem_used_max_chip": "",
    "npu_loaded_chips": "",
    "resource_samples": 0,
}


class ResourceMonitor:
    """后台采样器；start() 立即返回，stop() 返回聚合 dict（可直接 dict-update 进 CSV 行）。"""

    def __init__(self, interval_s: float = 2.0,
                 loaded_chip_mem_threshold_mb: int = 1024,
                 raw_log: Path | None = None):
        requested = float(interval_s)
        self.interval = max(MIN_INTERVAL_S, requested)
        if self.interval > requested:
            print(f"[monitor] interval {requested}s 太小，已上调到 "
                  f"{self.interval}s（最小 {MIN_INTERVAL_S}s）",
                  file=sys.stderr)
        self.threshold = loaded_chip_mem_threshold_mb
        self.raw_log = raw_log
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_samples: list[float] = []
        self._mem_samples: list[float] = []
        # 按 (chip_id, snapshot_idx) 平铺
        self._npu_samples: list[list[tuple[int, float, int, int]]] = []
        self._t0 = 0.0

    def start(self) -> None:
        self._stop_evt.clear()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        prev_cpu = _read_cpu_jiffies()
        raw_fh = None
        if self.raw_log is not None:
            try:
                raw_fh = open(self.raw_log, "a")
            except OSError as e:
                print(f"[monitor] 打开 raw_log {self.raw_log} 失败：{e}；"
                      f"继续采样但不写时序文件", file=sys.stderr)
        try:
            # 第一轮只刷新 prev_cpu（jiffies 差分需要前后两次读数），
            # 真正的 CPU 采样从下一轮开始；NPU/内存第一轮就采。
            warmup = True
            while not self._stop_evt.is_set():
                cpu_pct, prev_cpu = _cpu_percent(prev_cpu)
                mem_pct = _mem_percent()
                npu = _read_npu_smi()
                ts = time.time() - self._t0
                if cpu_pct is not None and not warmup:
                    self._cpu_samples.append(cpu_pct)
                if mem_pct is not None:
                    self._mem_samples.append(mem_pct)
                if npu:
                    self._npu_samples.append(npu)
                if raw_fh:
                    try:
                        raw_fh.write(
                            f"{ts:7.2f}  cpu={cpu_pct}  mem={mem_pct}  "
                            f"npu={npu}\n"
                        )
                        raw_fh.flush()
                    except OSError as e:
                        print(f"[monitor] raw_log 写入失败：{e}；停止时序记录",
                              file=sys.stderr)
                        try:
                            raw_fh.close()
                        except OSError:
                            pass
                        raw_fh = None
                warmup = False
                if self._stop_evt.wait(self.interval):
                    break
        finally:
            if raw_fh:
                try:
                    raw_fh.close()
                except OSError:
                    pass

    def stop(self) -> dict:
        self._stop_evt.set()
        if self._thread:
            # worker 可能正卡在 npu-smi（最长 _NPU_SMI_TIMEOUT_S）+ 处理一轮
            # 才看到 stop_evt，给出足够余量避免 join 提前返回时 _summarize
            # 还在和后台线程竞争追加。
            self._thread.join(timeout=self.interval + _NPU_SMI_TIMEOUT_S + 2.0)
            if self._thread.is_alive():
                print("[monitor] 后台线程超时未退出，结果可能不完整",
                      file=sys.stderr)
        return self._summarize()

    def _summarize(self) -> dict:
        out = dict(EMPTY_RESULT)
        out["resource_samples"] = len(self._cpu_samples)
        if self._cpu_samples:
            out["cpu_util_avg(%)"] = round(sum(self._cpu_samples) / len(self._cpu_samples), 2)
            out["cpu_util_max(%)"] = round(max(self._cpu_samples), 2)
        if self._mem_samples:
            out["mem_util_avg(%)"] = round(sum(self._mem_samples) / len(self._mem_samples), 2)
            out["mem_util_max(%)"] = round(max(self._mem_samples), 2)

        if self._npu_samples:
            # 识别"在用"的 chip：任一快照里 mem_used > threshold
            loaded: set[int] = set()
            for snap in self._npu_samples:
                for chip_id, _, mem_u, _ in snap:
                    if mem_u >= self.threshold:
                        loaded.add(chip_id)

            aicore_used: list[float] = []
            mem_peak: tuple[int, int] | None = None  # (mem_used, chip_id)
            for snap in self._npu_samples:
                for chip_id, aicore, mem_u, _ in snap:
                    if chip_id not in loaded:
                        continue
                    aicore_used.append(aicore)
                    if mem_peak is None or mem_u > mem_peak[0]:
                        mem_peak = (mem_u, chip_id)

            if loaded:
                out["npu_loaded_chips"] = ",".join(str(c) for c in sorted(loaded))
            if aicore_used:
                out["npu_aicore_avg(%)"] = round(sum(aicore_used) / len(aicore_used), 2)
                out["npu_aicore_max(%)"] = round(max(aicore_used), 2)
            if mem_peak is not None:
                out["npu_mem_used_max(MB)"] = mem_peak[0]
                out["npu_mem_used_max_chip"] = mem_peak[1]
        return out
