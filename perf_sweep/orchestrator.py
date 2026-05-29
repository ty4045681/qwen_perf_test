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
    RESOURCE_SAMPLE_INTERVAL_S,
    TEMPLATE_PATH,
    VLLM_HOST,
    VLLM_PORT,
)
from .plotter import plot_aggregated, plot_per_card, plot_per_deployment
from .resource_monitor import (
    EMPTY_RESULT as EMPTY_RES_FIELDS,
    ResourceMonitor,
    effective_interval,
)
from .result_parser import collect_results
from .vllm_server import (
    render_start_script,
    start_vllm,
    stop_vllm,
    wait_port_free,
    wait_until_ready,
)


PERF_FIELDS = [
    "deployment", "tp", "dp",
    "batch_size", "dataset", "num_prompt",
    "E2EL(ms)", "TTFT(ms)", "TPOT(ms)", "ITL(ms)", "OTT(token/s)",
]
# resource_monitor.EMPTY_RESULT 的 key 顺序就是我们想要的列顺序
RES_FIELDS = list(EMPTY_RES_FIELDS.keys())
CSV_FIELDS = PERF_FIELDS + RES_FIELDS + ["source_file"]

PER_CARD_FIELDS = [
    "deployment", "tp", "dp", "n_cards",
    "batch_size", "n_ott", "n_e2el",
    "OTT_mean(token/s)", "OTT_per_card_mean(token/s)",
    "E2EL_mean(ms)", "E2EL_max(ms)", "worst_E2EL_tier",
]


# ---------------------- CSV ----------------------
def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _empty_row(dep: Deployment, bs: int, lang: str, num_prompt: int) -> dict:
    row = {
        "deployment": dep.name, "tp": dep.tp, "dp": dep.dp,
        "batch_size": bs, "dataset": lang, "num_prompt": num_prompt,
        "E2EL(ms)": "", "TTFT(ms)": "", "TPOT(ms)": "",
        "ITL(ms)": "", "OTT(token/s)": "", "source_file": "",
    }
    row.update(EMPTY_RES_FIELDS)
    return row


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

    res_log_dir = dep_dir / "resource_samples"
    res_log_dir.mkdir(exist_ok=True)

    try:
        for bs in batch_sizes:
            patch_config(bs, args.max_out_len, args.config)
            num_prompt = bs * 2
            for lang, path in datasets.items():
                row = _empty_row(dep, bs, lang, num_prompt)

                # 起监控
                mon = None
                if not args.no_monitor:
                    raw_log = res_log_dir / f"bs{bs}_{lang}.tsv" if args.monitor_raw else None
                    mon = ResourceMonitor(
                        interval_s=args.monitor_interval,
                        raw_log=raw_log,
                    )
                    mon.start()

                since = time.time()
                try:
                    run_one(bs, path, args.work_dir)
                finally:
                    if mon is not None:
                        row.update(mon.stop())

                parsed = collect_results(args.work_dir, lang, since)
                if parsed is None:
                    print(f"       [warn] 未找到 {dep.name} bs={bs} {lang} 的 perf 结果")
                else:
                    metrics, src = parsed
                    row.update({k: ("" if v is None else v) for k, v in metrics.items()})
                    row["source_file"] = str(src)
                    print(f"       got: {metrics} <- {src}")
                # 总是打印监控行（即使 resource_samples=0），便于发现监控起
                # 起来但实际没采到数据的失败模式。
                if mon is not None:
                    n_samp = row.get("resource_samples", 0) or 0
                    if n_samp:
                        print(f"       res: cpu_avg={row['cpu_util_avg(%)']}% "
                              f"npu_aicore_avg={row['npu_aicore_avg(%)']}% "
                              f"npu_mem_max={row['npu_mem_used_max(MB)']}MB "
                              f"(samples={n_samp})")
                    else:
                        print(f"       res: [warn] 监控已启用但未采到样本 "
                              f"(samples=0)，可能 run_one 太短或线程异常退出")
                rows.append(row)
                _write_csv(rows, summary_csv, CSV_FIELDS)  # 增量落盘
    finally:
        if proc is not None:
            stop_vllm(proc)
            wait_port_free(VLLM_HOST, VLLM_PORT)


# ---------------------- 单卡吞吐汇总 ----------------------
def _build_per_card_rows(rows: list[dict], deployments: list[Deployment]) -> list[dict]:
    """跨数据集求 OTT 均值，再除以 tp*dp。每个 (dep, bs) 一行。"""
    dep_of = {d.name: d for d in deployments}
    bucket: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r.get("OTT(token/s)") in ("", None):
            continue
        if r["deployment"] not in dep_of:
            continue
        key = (r["deployment"], r["batch_size"])
        slot = bucket.setdefault(key, {"otts": [], "e2els": []})
        slot["otts"].append(float(r["OTT(token/s)"]))
        try:
            slot["e2els"].append(float(r["E2EL(ms)"]))
        except (TypeError, ValueError, KeyError):
            pass

    # 用 plotter 的 _worst_tier 保持视觉编码一致
    from .plotter import _worst_tier  # 局部 import 避免循环

    out: list[dict] = []
    for (dep_name, bs), slot in sorted(bucket.items()):
        dep = dep_of[dep_name]
        n_cards = dep.tp * dep.dp
        mean_ott = sum(slot["otts"]) / len(slot["otts"])
        n_e2el = len(slot["e2els"])
        if n_e2el:
            e2el_mean = sum(slot["e2els"]) / n_e2el
            e2el_max = max(slot["e2els"])
            worst_tier: int | str = -1
            for e in slot["e2els"]:
                worst_tier = max(int(worst_tier), _worst_tier(e / 1000.0))
        else:
            # 用 "" 把 "完全没有 E2EL 数据" 与 "都在最严门槛之内 (tier=-1)" 区分
            e2el_mean = ""
            e2el_max = ""
            worst_tier = ""
        out.append({
            "deployment": dep_name, "tp": dep.tp, "dp": dep.dp,
            "n_cards": n_cards,
            "batch_size": bs,
            "n_ott": len(slot["otts"]),
            "n_e2el": n_e2el,
            "OTT_mean(token/s)": round(mean_ott, 2),
            "OTT_per_card_mean(token/s)": round(mean_ott / n_cards, 2),
            "E2EL_mean(ms)": round(e2el_mean, 1) if isinstance(e2el_mean, float) else "",
            "E2EL_max(ms)": round(e2el_max, 1) if isinstance(e2el_max, float) else "",
            "worst_E2EL_tier": worst_tier,
        })
    return out


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
    if args.no_monitor:
        print(f"[orch] resource monitor = OFF")
    else:
        eff = effective_interval(args.monitor_interval)
        suffix = f" (requested {args.monitor_interval:.2f}s, clamped)" if eff != args.monitor_interval else ""
        print(f"[orch] resource monitor = ON (every {eff:.2f}s){suffix}")

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
            _write_csv(dep_rows, run_root / f"perf_summary_{dep.name}.csv", CSV_FIELDS)

    # 单卡平均吞吐 CSV
    per_card_rows = _build_per_card_rows(rows, deployments)
    if per_card_rows:
        _write_csv(per_card_rows, run_root / "per_card_summary.csv", PER_CARD_FIELDS)

    # 画图
    for dep in deployments:
        dep_rows = [r for r in rows if r["deployment"] == dep.name]
        if dep_rows:
            plot_per_deployment(dep_rows, dep,
                                run_root / f"perf_throughput_{dep.name}.png")
    if len(deployments) > 1:
        plot_aggregated(rows, deployments, run_root / "perf_throughput.png")
    # per_card_summary.csv 总是产出，per_card 图也总是产出（单部署时就一条曲线，
    # 仍能直观看到单卡吞吐随 batch_size 的变化）
    plot_per_card(rows, deployments, run_root / "per_card_throughput.png")

    print(f"\n[orch] done. {len(rows)} rows -> {summary_csv}")
    return 0
