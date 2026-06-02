"""合并多次同 sweep 的 perf_summary.csv，输出一份完整实验结果。

适用场景：同一组 tp/dp 部署被分多次 run_perf_sweep.py 跑完（典型是每次只覆盖
不同的 batch_sizes / cudagraph_capture_sizes），事后想拼成一份完整的
perf_summary.csv + per_card_summary.csv + 全套图。

合并语义：以 (tp, dp, batch_size, dataset) 为唯一键。按命令行给出的输入顺序，
后面的 run 覆盖前面的同键行（capture_sizes 不同的两次跑，认为后跑的更准），每条
被覆盖都打印告警。

注意：合并只读历史 CSV，不重跑 ais_bench，也不感知 cudagraph_capture_sizes —
它没有进过 CSV，仅影响数值本身。重建出的 Deployment 只用 tp/dp 和实际出现过的
batch_size 来驱动聚合与画图，cudagraph_capture_sizes 留空。
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from .deployments import DEFAULT_RUN_ROOT, DEPLOYMENTS, Deployment
from .orchestrator import (
    CSV_FIELDS,
    PER_CARD_FIELDS,
    _build_per_card_rows,
    _write_csv,
)
from .plotter import plot_aggregated, plot_per_card, plot_per_deployment


# 规范部署顺序：先按内置 DEPLOYMENTS 的顺序，方便聚合图布局与历史一致。
_CANONICAL_ORDER = {d.name: i for i, d in enumerate(DEPLOYMENTS)}


def _resolve_csv(p: Path) -> Path:
    """把一个输入路径解析成 perf_summary.csv 的实际位置。

    既接受 run 根目录（里面有 perf_summary.csv），也接受直接指向某个 csv。
    """
    if p.is_dir():
        csv_path = p / "perf_summary.csv"
        if not csv_path.is_file():
            raise SystemExit(f"目录里没有 perf_summary.csv: {p}")
        return csv_path
    if p.is_file():
        return p
    raise SystemExit(f"输入路径不存在: {p}")


def _coerce_row(row: dict) -> dict:
    """把 CSV 读回来的字符串行做最小必要的类型转换。

    batch_size / tp / dp 必须是 int —— 它们要做字典键、排序和 tp*dp 运算；
    其余指标/资源列保持字符串原样（下游一律用 float() 消费，空串当缺失）。
    """
    out = dict(row)
    for k in ("batch_size", "tp", "dp"):
        v = row.get(k, "")
        if v in ("", None):
            raise SystemExit(f"行缺少必需字段 {k!r}: {row}")
        out[k] = int(float(v))  # 容忍 "2.0" 这种写法
    return out


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [_coerce_row(r) for r in csv.DictReader(f)]


def _row_key(r: dict) -> tuple[int, int, int, str]:
    return (r["tp"], r["dp"], r["batch_size"], r.get("dataset", ""))


def _merge_rows(inputs: list[Path]) -> list[dict]:
    """按输入顺序合并，后跑覆盖先跑；返回保持插入顺序的行列表。"""
    merged: dict[tuple, dict] = {}
    origin: dict[tuple, str] = {}  # key -> 来自哪个 csv，用于告警
    for csv_path in inputs:
        rows = _load_rows(csv_path)
        print(f"[merge] {csv_path}: {len(rows)} 行")
        for r in rows:
            key = _row_key(r)
            if key in merged:
                print(f"[merge] [warn] 覆盖 tp{key[0]}_dp{key[1]} "
                      f"bs={key[2]} {key[3]}：{origin[key]} -> {csv_path}")
            merged[key] = r
            origin[key] = str(csv_path)
    return list(merged.values())


def _rebuild_deployments(rows: list[dict]) -> list[Deployment]:
    """从合并后的行里重建 Deployment（只为驱动聚合/画图）。

    batch_sizes = 该 (tp,dp) 实际出现过的 batch_size 升序去重；
    cudagraph_capture_sizes 留空（CSV 里没有这维信息）。
    """
    by_dep: dict[tuple[int, int], set[int]] = {}
    for r in rows:
        by_dep.setdefault((r["tp"], r["dp"]), set()).add(r["batch_size"])

    deps = [
        Deployment(tp=tp, dp=dp,
                   batch_sizes=sorted(bss),
                   cudagraph_capture_sizes=[])
        for (tp, dp), bss in by_dep.items()
    ]
    # 内置顺序优先，未知部署排在后面（按 tp、dp 兜底排序）
    deps.sort(key=lambda d: (_CANONICAL_ORDER.get(d.name, len(_CANONICAL_ORDER)),
                             d.tp, d.dp))
    return deps


def run_merge(inputs: list[str], out_dir: str | None) -> int:
    paths = [_resolve_csv(Path(p)) for p in inputs]
    if not paths:
        raise SystemExit("至少要给一个 run 目录或 perf_summary.csv")

    rows = _merge_rows(paths)
    if not rows:
        raise SystemExit("合并后没有任何行")
    deployments = _rebuild_deployments(rows)

    if out_dir:
        out = Path(out_dir)
    else:
        out = DEFAULT_RUN_ROOT / f"merged_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    print(f"[merge] out = {out}")
    print(f"[merge] deployments = {[d.name for d in deployments]}")
    print(f"[merge] {len(rows)} 行（去重后）")

    # 合并总表 + 分部署表（列与 orchestrator 完全一致）
    _write_csv(rows, out / "perf_summary.csv", CSV_FIELDS)
    for dep in deployments:
        dep_rows = [r for r in rows if r["deployment"] == dep.name]
        if dep_rows:
            _write_csv(dep_rows, out / f"perf_summary_{dep.name}.csv", CSV_FIELDS)

    # 单卡吞吐汇总
    per_card_rows = _build_per_card_rows(rows, deployments)
    if per_card_rows:
        _write_csv(per_card_rows, out / "per_card_summary.csv", PER_CARD_FIELDS)

    # 画图（与单次 sweep 同套产物）
    for dep in deployments:
        dep_rows = [r for r in rows if r["deployment"] == dep.name]
        if dep_rows:
            plot_per_deployment(dep_rows, dep,
                                out / f"perf_throughput_{dep.name}.png")
    if len(deployments) > 1:
        plot_aggregated(rows, deployments, out / "perf_throughput.png")
    plot_per_card(rows, deployments, out / "per_card_throughput.png")

    print(f"\n[merge] done. {len(rows)} rows -> {out / 'perf_summary.csv'}")
    return 0
