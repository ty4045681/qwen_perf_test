"""主流程：遍历 6 个部署，每个部署内遍历 batch_size × 数据集。"""
from __future__ import annotations

import csv
import time
from argparse import Namespace
from pathlib import Path

from .ais_bench_runner import patch_config, run_one
from .deployments import (
    AIS_CONFIG_PATH,
    AIS_WORK_DIR,
    DATASETS,
    DEPLOYMENTS,
    DEFAULT_RUN_ROOT,
    Deployment,
    MAX_OUT_LEN,
    MODEL_PATH,
    TEMPLATE_PATH,
    VLLM_HOST,
    VLLM_PORT,
)
from .plotter import plot_aggregated, plot_per_deployment
from .result_parser import collect_results
from .vllm_server import (
    render_start_script,
    start_vllm,
    stop_vllm,
    wait_port_free,
    wait_until_ready,
)


CSV_FIELDS = [
    "deployment", "tp", "dp",
    "batch_size", "dataset", "num_prompt",
    "E2EL(ms)", "TTFT(ms)", "TPOT(ms)", "ITL(ms)", "OTT(token/s)",
    "source_file",
]


# ---------------------- CSV ----------------------
def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def _empty_row(dep: Deployment, bs: int, lang: str, num_prompt: int) -> dict:
    return {
        "deployment": dep.name, "tp": dep.tp, "dp": dep.dp,
        "batch_size": bs, "dataset": lang, "num_prompt": num_prompt,
        "E2EL(ms)": "", "TTFT(ms)": "", "TPOT(ms)": "",
        "ITL(ms)": "", "OTT(token/s)": "", "source_file": "",
    }


# ---------------------- 选择/过滤 ----------------------
def _select_deployments(args: Namespace) -> list[Deployment]:
    if not args.only:
        return list(DEPLOYMENTS)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    sel = [d for d in DEPLOYMENTS if d.name in wanted]
    missing = wanted - {d.name for d in sel}
    if missing:
        raise SystemExit(f"未知的部署名: {sorted(missing)}；可选: "
                         f"{[d.name for d in DEPLOYMENTS]}")
    return sel


def _select_datasets(args: Namespace) -> dict[str, str]:
    if not args.datasets:
        return dict(DATASETS)
    wanted = [x.strip() for x in args.datasets.split(",") if x.strip()]
    sel = {k: DATASETS[k] for k in wanted if k in DATASETS}
    missing = [k for k in wanted if k not in DATASETS]
    if missing:
        raise SystemExit(f"未知的数据集: {missing}；可选: {list(DATASETS)}")
    return sel


def _effective_batch_sizes(dep: Deployment, args: Namespace) -> list[int]:
    if not args.batch_sizes:
        return list(dep.batch_sizes)
    return [int(x) for x in args.batch_sizes.split(",") if x.strip()]


# ---------------------- 单部署 sweep ----------------------
def _sweep_deployment(dep: Deployment, dep_dir: Path, datasets: dict[str, str],
                      batch_sizes: list[int], args: Namespace,
                      rows: list[dict], summary_csv: Path) -> None:
    """跑完单个部署的全部 (batch_size × dataset) 组合，结果 append 到 rows。"""
    proc = None
    if not args.skip_launch:
        script = render_start_script(dep, dep_dir, TEMPLATE_PATH, MODEL_PATH, VLLM_PORT)
        print(f"[orch] {dep.name}: start script -> {script}")
        if args.dry_run:
            print(f"[orch] dry-run: 不启动 vllm")
            return
        proc = start_vllm(script, dep_dir / "vllm.log")
        if not wait_until_ready(VLLM_HOST, VLLM_PORT, proc,
                                timeout=args.ready_timeout):
            print(f"[orch] {dep.name}: 未就绪，跳过本部署")
            stop_vllm(proc)
            wait_port_free(VLLM_HOST, VLLM_PORT)
            return
    else:
        print(f"[orch] {dep.name}: skip-launch，假定 vllm 已在 {VLLM_HOST}:{VLLM_PORT}")

    try:
        for bs in batch_sizes:
            patch_config(bs, args.max_out_len, args.config)
            num_prompt = bs * 2
            for lang, path in datasets.items():
                since = time.time()
                run_one(bs, path, args.work_dir)
                parsed = collect_results(args.work_dir, lang, since)
                if parsed is None:
                    print(f"       [warn] 未找到 {dep.name} bs={bs} {lang} 的 perf 结果")
                    rows.append(_empty_row(dep, bs, lang, num_prompt))
                else:
                    metrics, src = parsed
                    row = _empty_row(dep, bs, lang, num_prompt)
                    row.update({k: ("" if v is None else v) for k, v in metrics.items()})
                    row["source_file"] = str(src)
                    rows.append(row)
                    print(f"       got: {metrics} <- {src}")
                _write_csv(rows, summary_csv)  # 增量落盘
    finally:
        if proc is not None:
            stop_vllm(proc)
            wait_port_free(VLLM_HOST, VLLM_PORT)


# ---------------------- 入口 ----------------------
def run(args: Namespace) -> int:
    deployments = _select_deployments(args)
    datasets = _select_datasets(args)

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        run_root = DEFAULT_RUN_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    run_root = run_root.resolve()
    print(f"[orch] run_root = {run_root}")
    print(f"[orch] deployments = {[d.name for d in deployments]}")
    print(f"[orch] datasets    = {list(datasets)}")

    summary_csv = run_root / "perf_summary.csv"
    rows: list[dict] = []

    for dep in deployments:
        dep_dir = run_root / dep.name
        dep_dir.mkdir(parents=True, exist_ok=True)
        batch_sizes = _effective_batch_sizes(dep, args)
        print(f"\n[orch] ===== {dep.name} ===== batch_sizes={batch_sizes}")
        _sweep_deployment(dep, dep_dir, datasets, batch_sizes, args,
                          rows, summary_csv)

    if args.dry_run:
        print("[orch] dry-run 完成")
        return 0

    # 分部署 CSV
    for dep in deployments:
        dep_rows = [r for r in rows if r["deployment"] == dep.name]
        if dep_rows:
            _write_csv(dep_rows, run_root / f"perf_summary_{dep.name}.csv")

    # 画图
    for dep in deployments:
        dep_rows = [r for r in rows if r["deployment"] == dep.name]
        if dep_rows:
            plot_per_deployment(dep_rows, dep,
                                run_root / f"perf_throughput_{dep.name}.png")
    if len(deployments) > 1:
        plot_aggregated(rows, deployments, run_root / "perf_throughput.png")

    print(f"\n[orch] done. {len(rows)} rows -> {summary_csv}")
    return 0
