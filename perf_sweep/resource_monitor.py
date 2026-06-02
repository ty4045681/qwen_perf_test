"""资源监控：CPU / 系统内存 / NPU AI core / NPU 显存。

后台线程定期采样；run_one 期间 start() → 跑测试 → stop() → 拿聚合结果合并到 CSV 行。

数据源：
  - CPU%   : /proc/stat 两次采样差分（不依赖 psutil）
  - 内存%  : /proc/meminfo (MemTotal - MemAvailable) / MemTotal
  - NPU    : npu-smi info 解析，从表头行取 NPU 编号(0–7)、从指标行取 AICore(%) 与
             HBM-Usage(MB)；24.x 版同时有 Memory-Usage/HBM 两列，取 HBM（设备显存）

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
# npu-smi info 里每个 NPU 占 2 行。两种已知格式：
#   910B3（训练卡，HBM）：
#     | 0     910B3 | OK           | 96.7  50  0 / 0            |  <- 表头：NPU 编号 + 名称
#     | 0           | 0000:C1:00.0 | 0     0 / 0   3401 / 65536 |  <- 指标：Chip + Bus-Id + 指标
#   310P3（推理卡，LPDDR）：
#     | 0     310P3 | OK           | NA    32   788 / 788       |  <- 表头：NPU 编号 + 名称
#     | 0     0     | 0000:01:00.0 | 0     3497 / 44213         |  <- 指标：Chip + Device + Bus-Id + 指标
# 差异：310P3 指标行第一格是 "Chip Device" 两个数字（910B3 只有 Chip 一个），
# 所以 _NPU_METRIC_RE 首格用 \d+(?:\s+\d+)? 兼容两种。NPU 编号一律从表头行取
# （指标行的 Chip 恒为 0，单靠它会让所有卡塌缩成同一个 id）。
#
# 显存列在不同版本/型号下不一样：
#   - 单对 "Memory-Usage(MB)  x / y"：这一对就是设备显存（310P3 LPDDR / 旧版 HBM）；
#   - 两对 "Memory-Usage(MB) 0 / 0  HBM-Usage(MB) 3401 / 65536"：910B3 24.x，
#     第一对（DDR）恒为 0/0，真正的设备显存是第二对 HBM。
# 所以出现第二对就用第二对，否则用第一对。
#
# 解析顺序很关键：310P3 指标行的两数字首格 "| 0  0 |" 也会被 _NPU_HDR_RE 命中，
# 因此循环里必须先用带 Bus-Id 锚点的 _NPU_METRIC_RE 判定指标行、匹配上就 continue，
# 不让它落到表头分支被当成表头吞掉（这正是 310P3 上 NPU 指标全空的根因）。
_NPU_HDR_RE = re.compile(r"^\s*\|\s*(\d+)\s+\S+\s*\|")
_NPU_METRIC_RE = re.compile(
    r"\|\s*\d+(?:\s+\d+)?\s*\|\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f:.]+\s*\|\s*"
    r"([\d.]+)\s+(\d+)\s*/\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?"
)
# 输出底部进程表行形如：
#   | 0       0                 | 4038460       | VLLMWorker               | 118       |
# 即 "| NPU Chip | PID | 进程名 | 进程显存 |"。第二列是纯数字 PID（指标行那里是
# 带冒号的 Bus-Id），靠这点把进程行和指标行区分开。"No running processes..." 行
# 第一列不是数字，自然不匹配。
_NPU_PROC_RE = re.compile(r"^\s*\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|\s*(\S+)")


def _read_npu_smi() -> tuple[list[tuple[int, float, int, int]], set[int]]:
    """返回 (rows, proc_chips)。
      rows       : 每个 NPU 的 (npu_id, aicore_pct, hbm_used_MB, hbm_total_MB)
      proc_chips : 进程表里挂着进程（vLLM 等）的 NPU 编号集合——这是"这张卡
                   正被本次部署占用"的权威信号，与 HBM 用量多少无关。
    npu-smi 不可用或解析失败返回 ([], set())。"""
    try:
        proc = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True,
            timeout=_NPU_SMI_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return [], set()
    if proc.returncode != 0:
        return [], set()
    rows: list[tuple[int, float, int, int]] = []
    proc_chips: set[int] = set()
    cur_npu: int | None = None
    fallback = 0  # 万一没解析到表头行，用递增序号兜底，避免全部塌缩成同一个 id
    for line in proc.stdout.splitlines():
        # 进程行要先于表头判定：进程行的第一格 "0  0" 也会被 _NPU_HDR_RE 命中，
        # 但它带的第二个数字是 PID，用 _NPU_PROC_RE 精确识别。
        pm = _NPU_PROC_RE.search(line)
        if pm:
            proc_chips.add(int(pm.group(1)))
            continue
        # 指标行要先于表头判定：310P3 指标行首格 "0  0" 也会被 _NPU_HDR_RE 命中，
        # 但指标行第二格是 Bus-Id，_NPU_METRIC_RE 靠它精确识别，匹配上即 continue。
        m = _NPU_METRIC_RE.search(line)
        if m:
            g = m.groups()
            try:
                aicore = float(g[0])
                # 有第二对 (HBM) 就用 HBM，否则用第一对
                if g[3] is not None:
                    mem_used, mem_total = int(g[3]), int(g[4])
                else:
                    mem_used, mem_total = int(g[1]), int(g[2])
            except (ValueError, TypeError):
                continue
            npu_id = cur_npu if cur_npu is not None else fallback
            rows.append((npu_id, aicore, mem_used, mem_total))
            cur_npu = None
            fallback += 1
            continue
        hdr = _NPU_HDR_RE.match(line)
        if hdr:
            cur_npu = int(hdr.group(1))
            continue
    return rows, proc_chips


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
                 loaded_chip_mem_threshold_mb: int = 8192,
                 raw_log: Path | None = None):
        # "在用卡" = 进程表（挂着本次部署 worker 的卡）与显存阈值（显存越过基线的
        # 卡）的交集，详见 _summarize。本阈值即交集里的显存判据：显存用量越过此值
        # 才算真正灌入了权重。910B3 读 HBM-Usage / 310P3 读 Memory-Usage，空载基线
        # 约 2–3GB，默认 8192MB（8GB）足以越过基线。
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
        # 整个采样窗口里出现过 vLLM 进程的 NPU 编号并集（权威的"在用卡"信号）
        self._npu_proc_chips: set[int] = set()
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
                npu, proc_chips = _read_npu_smi()
                ts = time.time() - self._t0
                if cpu_pct is not None and not warmup:
                    self._cpu_samples.append(cpu_pct)
                if mem_pct is not None:
                    self._mem_samples.append(mem_pct)
                if npu:
                    self._npu_samples.append(npu)
                self._npu_proc_chips |= proc_chips
                if raw_fh:
                    try:
                        raw_fh.write(
                            f"{ts:7.2f}  cpu={cpu_pct}  mem={mem_pct}  "
                            f"proc_chips={sorted(proc_chips)}  npu={npu}\n"
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
            # 识别"在用"的 chip：进程表（哪张卡挂着本次部署的 worker）与显存阈值
            # （哪张卡显存越过基线）取交集——既挂着进程、显存又确实灌满了才算在用。
            # 这样共享机器上别的容器在某张卡留下的小进程（有进程但显存很低）不会被
            # 误算进来。
            proc_chips: set[int] = set(self._npu_proc_chips)
            thresh_chips: set[int] = set()
            for snap in self._npu_samples:
                for chip_id, _, mem_u, _ in snap:
                    if mem_u >= self.threshold:
                        thresh_chips.add(chip_id)
            # 退化兜底：两路信号都在就取交集；交集为空或某一路缺失时，按
            #   交集 → 纯显存阈值 → 纯进程表 的优先级回退，避免直接判成"无在用卡"
            # 而丢掉全部 NPU 指标。
            loaded: set[int] = (proc_chips & thresh_chips) or thresh_chips or proc_chips

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
