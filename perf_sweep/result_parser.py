"""解析 ais_bench 输出。

ais_bench 每次调用会在 work_dir 下生成一个时间戳目录，结构如：
  {work_dir}/{YYYYMMDD_HHMMSS}/performances/vllm-api-stream-chat/{lang}.csv
  {work_dir}/{YYYYMMDD_HHMMSS}/performances/vllm-api-stream-chat/{lang}.json

CSV 行格式：
  Performance Parameters,Stage,Average,Min,Max,Median,P75,P90,P99,N
  E2EL,total,242379.7 ms,...
  TTFT,total,91206.0 ms,...
  TPOT,total,759.7 ms,...
  ITL,total,2519.1 ms,...
JSON 字段：
  "Output Token Throughput": {"total": "125.3534 token/s"}
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PERF_SUBDIR = Path("performances") / "vllm-api-stream-chat"


def _strip_unit(s) -> float | None:
    """'242379.7 ms' -> 242379.7"""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None


def parse_csv_metrics(csv_path: Path) -> dict:
    """从 ais_bench 的 perf CSV 中取 E2EL/TTFT/TPOT/ITL 的 Average 列。"""
    out = {"E2EL(ms)": None, "TTFT(ms)": None, "TPOT(ms)": None, "ITL(ms)": None}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return out
        # 找 'Average' 列下标（容错大小写/空格）
        try:
            avg_idx = next(i for i, h in enumerate(header)
                           if h.strip().lower() == "average")
        except StopIteration:
            avg_idx = 2  # 按观察到的格式回退
        key_map = {"E2EL": "E2EL(ms)", "TTFT": "TTFT(ms)",
                   "TPOT": "TPOT(ms)", "ITL": "ITL(ms)"}
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if name in key_map and len(row) > avg_idx:
                out[key_map[name]] = _strip_unit(row[avg_idx])
    return out


def parse_json_ott(json_path: Path) -> float | None:
    """从 ais_bench 的 perf JSON 中取 Output Token Throughput。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    v = data.get("Output Token Throughput")
    if isinstance(v, dict):
        v = v.get("total")
    return _strip_unit(v)


def collect_results(work_dir: str, dataset_name: str, since_ts: float):
    """
    找出 since_ts 之后新生成、包含 {dataset_name}.csv 的时间戳目录，
    解析指标并返回 (metrics_dict, source_csv_path)。
    """
    root = Path(work_dir)
    if not root.exists():
        return None

    candidates = []
    for ts_dir in root.iterdir():
        if not ts_dir.is_dir():
            continue
        csv_path = ts_dir / PERF_SUBDIR / f"{dataset_name}.csv"
        json_path = ts_dir / PERF_SUBDIR / f"{dataset_name}.json"
        if not csv_path.exists():
            continue
        try:
            mtime = csv_path.stat().st_mtime
        except OSError:
            continue
        if mtime < since_ts - 5:
            continue
        candidates.append((mtime, csv_path, json_path))

    if not candidates:
        return None

    # 取 since_ts 之后最早的那次
    candidates.sort()
    _, csv_path, json_path = candidates[0]

    metrics = parse_csv_metrics(csv_path)
    metrics["OTT(token/s)"] = parse_json_ott(json_path) if json_path.exists() else None
    return metrics, csv_path
