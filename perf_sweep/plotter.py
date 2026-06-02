"""画图：单部署双面板 + 6 部署聚合 + 单卡吞吐对比。"""
from __future__ import annotations

from pathlib import Path

from .deployments import Deployment, E2EL_BUDGETS


# E2EL 门槛的视觉编码。前两档手工调色（与历史一致）；多于 2 档时从 colormap
# 取色，保证每档颜色都唯一可辨。约定 index 越大 = 越严重 = 越靠后绘制覆盖。
_TIER_STYLES_BASE = [
    {"overlay_face": "none", "overlay_edge": "darkorange",
     "line_color": "darkorange", "line_style": "--", "z": 4},
    {"overlay_face": "red", "overlay_edge": "red",
     "line_color": "red", "line_style": "--", "z": 6},
]


def _budget_style(idx: int) -> dict:
    if idx < len(_TIER_STYLES_BASE):
        return _TIER_STYLES_BASE[idx]
    # 第 3 档及以后：用 matplotlib 的 'Reds' colormap 继续向深红推
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("Reds")
        # 0.6 ~ 1.0 的深红区间，避免和前两档混淆
        n_extra = max(1, len(E2EL_BUDGETS) - len(_TIER_STYLES_BASE))
        t = 0.6 + 0.4 * ((idx - len(_TIER_STYLES_BASE) + 1) / n_extra)
        color = cmap(min(1.0, t))
    except Exception:
        color = "darkred"
    return {"overlay_face": color, "overlay_edge": color,
            "line_color": color, "line_style": "--", "z": 6 + idx}


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[plot] matplotlib 未安装，跳过画图。pip install matplotlib")
        return None


def _group_by_lang(rows: list[dict]) -> dict:
    by_lang: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("OTT(token/s)") in ("", None):
            continue
        by_lang.setdefault(r["dataset"], []).append(r)
    for lst in by_lang.values():
        lst.sort(key=lambda x: x["batch_size"])
    return by_lang


def _e2el_seconds(r: dict) -> float | None:
    try:
        return float(r["E2EL(ms)"]) / 1000.0
    except (TypeError, ValueError, KeyError):
        return None


def _worst_tier(e2el_s: float | None) -> int:
    """返回 e2el 跨过的最高门槛位置；都没跨返回 -1。"""
    if e2el_s is None:
        return -1
    worst = -1
    for i, b in enumerate(E2EL_BUDGETS):
        if e2el_s > b:
            worst = i
    return worst


def _draw_throughput(ax, by_lang: dict, lang_color: dict,
                     *, with_legend: bool, title: str | None) -> None:
    """画 Throughput vs Concurrency 曲线 + 多档 E2EL 超预算标记。"""
    for lang, lst in sorted(by_lang.items()):
        xs = [r["batch_size"] for r in lst]
        ys = [float(r["OTT(token/s)"]) for r in lst]
        ax.plot(xs, ys, marker="o", color=lang_color[lang],
                label=lang, linewidth=1.8)
        for r in lst:
            tier = _worst_tier(_e2el_seconds(r))
            if tier < 0:
                continue
            s = _budget_style(tier)
            ax.scatter([r["batch_size"]], [float(r["OTT(token/s)"])],
                       s=140, facecolors=s["overlay_face"], edgecolors=s["overlay_edge"],
                       linewidths=1.8, zorder=s["z"])
    ax.set_ylabel("Output Throughput (token/s)")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if with_legend and by_lang:
        ax.legend(title="dataset", loc="upper left", ncol=max(1, len(by_lang)))


def _draw_e2el(ax, by_lang: dict, lang_color: dict, all_bs: list[int]) -> None:
    """下面板：E2EL vs Concurrency + 各档预算虚线。"""
    for lang, lst in sorted(by_lang.items()):
        xs, ys = [], []
        for r in lst:
            e2el_s = _e2el_seconds(r)
            if e2el_s is None:
                continue
            xs.append(r["batch_size"])
            ys.append(e2el_s)
        if xs:
            ax.plot(xs, ys, marker="o", color=lang_color[lang],
                    linewidth=1.8, label=lang)
    for i, b in enumerate(E2EL_BUDGETS):
        s = _budget_style(i)
        ax.axhline(b, color=s["line_color"], linestyle=s["line_style"],
                   linewidth=1.2, label=f"E2EL budget = {b:.0f}s")
    ax.set_xlabel("Concurrency (batch_size)")
    ax.set_ylabel("E2E Latency (s)")
    if all_bs:
        ax.set_xticks(all_bs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=max(1, len(by_lang) + len(E2EL_BUDGETS)), fontsize=8)


def _budget_legend_handles(plt):
    handles = []
    for i, b in enumerate(E2EL_BUDGETS):
        s = _budget_style(i)
        handles.append(plt.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=(s["overlay_face"] if s["overlay_face"] != "none" else "white"),
            markeredgecolor=s["overlay_edge"],
            markersize=10, markeredgewidth=1.8,
            label=f"E2EL > {b:.0f}s",
        ))
    return handles


def plot_per_deployment(rows: list[dict], dep: Deployment, out_path: Path) -> None:
    """单部署双面板图（throughput + e2el）。"""
    plt = _try_import_matplotlib()
    if plt is None:
        return
    by_lang = _group_by_lang(rows)
    if not by_lang:
        print(f"[plot] {dep.name} 无有效数据，跳过")
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )
    cmap = plt.get_cmap("tab10")
    lang_color = {lang: cmap(i) for i, lang in enumerate(sorted(by_lang.keys()))}

    budget_desc = ", ".join(
        f"{_budget_style(i)['overlay_edge']} = E2EL>{b:.0f}s"
        for i, b in enumerate(E2EL_BUDGETS)
    )
    _draw_throughput(
        ax_top, by_lang, lang_color,
        with_legend=True,
        title=(f"910B3 Qwen3.6-35B-A3B-w8a8 [{dep.name}]: "
               f"Throughput & E2E Latency vs Concurrency\n"
               f"({budget_desc})"),
    )
    all_bs = sorted({r["batch_size"] for lst in by_lang.values() for r in lst})
    _draw_e2el(ax_bot, by_lang, lang_color, all_bs)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


# ---------------------- 资源利用率（CPU/内存/NPU） ----------------------
# 每个指标：CSV 里的 avg/max 列、图例标签、颜色、文件名后缀、单位。
# avg 为 None 表示该指标只有 max（如 NPU 显存只采峰值）。
_RES_PCT_METRICS = [
    {"label": "CPU", "color": "tab:blue", "file": "cpu",
     "avg": "cpu_util_avg(%)", "max": "cpu_util_max(%)", "unit": "%"},
    {"label": "NPU AICore", "color": "tab:red", "file": "npu_aicore",
     "avg": "npu_aicore_avg(%)", "max": "npu_aicore_max(%)", "unit": "%"},
    {"label": "Memory", "color": "tab:green", "file": "mem",
     "avg": "mem_util_avg(%)", "max": "mem_util_max(%)", "unit": "%"},
]
_RES_MEM_METRIC = {
    "label": "NPU Mem Used", "color": "tab:purple", "file": "npu_mem",
    "avg": None, "max": "npu_mem_used_max(MB)", "unit": "MB",
}


def _float_or_none(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _resource_series(rows: list[dict], deployments: list[Deployment],
                     avg_col: str | None, max_col: str) -> dict:
    """{dep_name: [(batch_size, avg, max), ...]}，按 batch_size 升序。

    资源是整机量、与数据集基本无关，所以跨数据集归约：
      avg = 各数据集 avg 列的均值；max = 各数据集 max 列的最大值。
    某点两者都缺则跳过；avg_col 为 None（只有 max 的指标）时 avg 恒为 None。
    """
    dep_names = {d.name for d in deployments}
    bucket: dict[tuple[str, int], dict] = {}
    for r in rows:
        dep = r.get("deployment")
        if dep not in dep_names:
            continue
        key = (dep, r["batch_size"])
        slot = bucket.setdefault(key, {"avgs": [], "maxs": []})
        if avg_col:
            a = _float_or_none(r.get(avg_col))
            if a is not None:
                slot["avgs"].append(a)
        m = _float_or_none(r.get(max_col))
        if m is not None:
            slot["maxs"].append(m)

    out: dict[str, list[tuple[int, float | None, float | None]]] = {
        d.name: [] for d in deployments}
    for (dep, bs), slot in bucket.items():
        avg = sum(slot["avgs"]) / len(slot["avgs"]) if slot["avgs"] else None
        mx = max(slot["maxs"]) if slot["maxs"] else None
        if avg is None and mx is None:
            continue
        out[dep].append((bs, avg, mx))
    for v in out.values():
        v.sort()
    return out


def _avg_max_style_handles(plt):
    """两个线型图例项：实线=avg，虚线=max（黑色，仅示意线型）。"""
    return [
        plt.Line2D([0], [0], color="black", linewidth=1.8, label="avg"),
        plt.Line2D([0], [0], color="black", linewidth=1.4, linestyle="--", label="max"),
    ]


def plot_resource_aggregated(rows: list[dict], deployments: list[Deployment],
                             out_path: Path) -> None:
    """聚合网格：每个部署一个子图，叠画 CPU%/NPU AICore%/内存% 随 batch_size。

    类比 plot_aggregated。每个指标一种颜色，avg 实线、max 虚线；% 同轴 0-100。
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return

    series_by_metric = {
        m["label"]: _resource_series(rows, deployments, m["avg"], m["max"])
        for m in _RES_PCT_METRICS
    }
    if not any(any(s.values()) for s in series_by_metric.values()):
        print("[plot] resource: 无资源数据（监控可能关闭或未采到），跳过聚合图")
        return

    n = len(deployments)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows),
                             sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    all_bs = sorted({bs for s in series_by_metric.values()
                     for pts in s.values() for (bs, _, _) in pts})

    for ax, dep in zip(axes, deployments):
        drew = False
        for m in _RES_PCT_METRICS:
            pts = series_by_metric[m["label"]].get(dep.name, [])
            if not pts:
                continue
            drew = True
            avg_pts = [(bs, a) for (bs, a, _) in pts if a is not None]
            max_pts = [(bs, mx) for (bs, _, mx) in pts if mx is not None]
            if avg_pts:
                ax.plot([p[0] for p in avg_pts], [p[1] for p in avg_pts],
                        marker="o", color=m["color"], linewidth=1.8)
            if max_pts:
                ax.plot([p[0] for p in max_pts], [p[1] for p in max_pts],
                        marker="x", color=m["color"], linewidth=1.4,
                        linestyle="--", alpha=0.7)
        ax.set_title(dep.name if drew else f"{dep.name}  (no data)")
        ax.set_xlabel("Concurrency (batch_size)")
        ax.set_ylabel("Utilization (%)")
        ax.set_ylim(0, 105)
        if all_bs:
            ax.set_xticks(all_bs)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(deployments):]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], color=m["color"], linewidth=1.8,
                          label=m["label"]) for m in _RES_PCT_METRICS]
    handles.extend(_avg_max_style_handles(plt))
    fig.legend(handles=handles, loc="upper center",
               ncol=len(handles), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("910B3 Qwen3.6-35B-A3B-w8a8: Resource utilization vs concurrency "
                 "across deployments", y=1.06, fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


def plot_resource_metric(rows: list[dict], deployments: list[Deployment],
                         metric: dict, out_path: Path) -> None:
    """单指标跨部署对比：每个部署一条曲线，avg 实线 + max 虚线。

    类比 plot_per_card。% 指标固定 0-100 轴；MB 指标（如 NPU 显存）自适应。
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return

    series = _resource_series(rows, deployments, metric["avg"], metric["max"])
    if not any(series.values()):
        print(f"[plot] resource[{metric['label']}]: 无数据，跳过")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    dep_color = {d.name: cmap(i) for i, d in enumerate(deployments)}

    for dep in deployments:
        pts = series.get(dep.name, [])
        if not pts:
            continue
        color = dep_color[dep.name]
        avg_pts = [(bs, a) for (bs, a, _) in pts if a is not None]
        max_pts = [(bs, mx) for (bs, _, mx) in pts if mx is not None]
        label = f"{dep.name} ({dep.tp * dep.dp} cards)"
        if avg_pts:
            ax.plot([p[0] for p in avg_pts], [p[1] for p in avg_pts],
                    marker="o", linewidth=1.8, color=color, label=label)
        if max_pts:
            # 没有 avg 线时（如 NPU 显存）用实线承载图例标签，否则 max 走虚线
            ls, lbl = ("--", None) if avg_pts else ("-", label)
            ax.plot([p[0] for p in max_pts], [p[1] for p in max_pts],
                    marker="x", linewidth=1.4, linestyle=ls,
                    color=color, alpha=0.7, label=lbl)

    ax.set_xlabel("Concurrency (batch_size)")
    ax.set_ylabel(f"{metric['label']} ({metric['unit']})")
    if metric["unit"] == "%":
        ax.set_ylim(0, 105)
    all_bs = sorted({bs for pts in series.values() for (bs, _, _) in pts})
    if all_bs:
        ax.set_xticks(all_bs)
    has_avg = metric["avg"] is not None
    detail = "avg solid + max dashed" if has_avg else "peak only"
    ax.set_title(
        f"910B3 Qwen3.6-35B-A3B-w8a8: {metric['label']} utilization vs concurrency\n"
        f"(mean across datasets; {detail})"
    )
    ax.grid(True, alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    if has_avg:
        handles = list(handles) + _avg_max_style_handles(plt)
    ax.legend(handles=handles, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


def plot_resources(rows: list[dict], deployments: list[Deployment],
                   out_dir: Path) -> None:
    """出全套资源图：聚合网格 + 每个 % 指标跨部署对比 + NPU 显存(MB) 单独图。"""
    plot_resource_aggregated(rows, deployments, out_dir / "resource_util.png")
    for m in _RES_PCT_METRICS:
        plot_resource_metric(rows, deployments, m,
                             out_dir / f"resource_{m['file']}.png")
    plot_resource_metric(rows, deployments, _RES_MEM_METRIC,
                         out_dir / f"resource_{_RES_MEM_METRIC['file']}.png")


def plot_aggregated(rows: list[dict], deployments: list[Deployment],
                    out_path: Path) -> None:
    """6 部署聚合图：2×3 子图，每格一个部署的 throughput 面板。共享 y 轴比例。"""
    plt = _try_import_matplotlib()
    if plt is None:
        return

    by_dep: dict[str, list[dict]] = {d.name: [] for d in deployments}
    for r in rows:
        if r.get("deployment") in by_dep:
            by_dep[r["deployment"]].append(r)

    all_langs = sorted({r["dataset"] for r in rows
                        if r.get("OTT(token/s)") not in ("", None)})
    cmap = plt.get_cmap("tab10")
    lang_color = {lang: cmap(i) for i, lang in enumerate(all_langs)}

    n = len(deployments)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows),
                             sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, dep in zip(axes, deployments):
        by_lang = _group_by_lang(by_dep.get(dep.name, []))
        if not by_lang:
            ax.set_title(f"{dep.name}  (no data)")
            ax.grid(True, alpha=0.3)
            continue
        _draw_throughput(ax, by_lang, lang_color, with_legend=False, title=dep.name)
        ax.set_xlabel("Concurrency (batch_size)")

    for ax in axes[len(deployments):]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], marker="o", color=lang_color[lang],
                          label=lang, linewidth=1.8)
               for lang in all_langs]
    handles.extend(_budget_legend_handles(plt))
    fig.legend(handles=handles, loc="upper center",
               ncol=len(all_langs) + len(E2EL_BUDGETS),
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("910B3 Qwen3.6-35B-A3B-w8a8: Throughput vs Concurrency across deployments",
                 y=1.06, fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


# ---------------------- 单卡吞吐对比 ----------------------
def _per_card_aggregate(rows: list[dict], deployments: list[Deployment]) -> dict:
    """返回 {dep_name: [(batch_size, per_card_ott, worst_tier_across_langs), ...]}。

    per_card_ott = mean(OTT across langs) / (tp*dp)
    worst_tier = 该 (dep, bs) 跨数据集里出现过的最高 E2EL 门槛档位。
    """
    bucket: dict[tuple[str, int], dict] = {}
    dep_of = {d.name: d for d in deployments}
    for r in rows:
        if r.get("OTT(token/s)") in ("", None):
            continue
        dep_name = r["deployment"]
        if dep_name not in dep_of:
            continue
        key = (dep_name, r["batch_size"])
        slot = bucket.setdefault(key, {"otts": [], "worst_tier": -1})
        slot["otts"].append(float(r["OTT(token/s)"]))
        slot["worst_tier"] = max(slot["worst_tier"], _worst_tier(_e2el_seconds(r)))

    out: dict[str, list[tuple[int, float, int]]] = {d.name: [] for d in deployments}
    for (dep_name, bs), slot in bucket.items():
        dep = dep_of[dep_name]
        mean_ott = sum(slot["otts"]) / len(slot["otts"])
        per_card = mean_ott / (dep.tp * dep.dp)
        out[dep_name].append((bs, per_card, slot["worst_tier"]))
    for v in out.values():
        v.sort()
    return out


def plot_per_card(rows: list[dict], deployments: list[Deployment], out_path: Path) -> None:
    """每个部署一条曲线：横轴 batch_size，纵轴跨数据集平均后的单卡吞吐。"""
    plt = _try_import_matplotlib()
    if plt is None:
        return

    series = _per_card_aggregate(rows, deployments)
    if not any(series.values()):
        print("[plot] per-card: 无数据")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    dep_color = {d.name: cmap(i) for i, d in enumerate(deployments)}

    for dep in deployments:
        pts = series.get(dep.name, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.8,
                color=dep_color[dep.name],
                label=f"{dep.name} ({dep.tp * dep.dp} cards)")
        for bs, per_card, tier in pts:
            if tier < 0:
                continue
            s = _budget_style(tier)
            ax.scatter([bs], [per_card], s=140,
                       facecolors=s["overlay_face"], edgecolors=s["overlay_edge"],
                       linewidths=1.8, zorder=s["z"])

    ax.set_xlabel("Concurrency (batch_size)")
    ax.set_ylabel("Per-card Output Throughput (token/s)")
    ax.set_title(
        "910B3 Qwen3.6-35B-A3B-w8a8: Per-card throughput vs concurrency\n"
        "(mean across 5 datasets; overlay = worst E2EL tier seen in those datasets)"
    )
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles = list(handles) + _budget_legend_handles(plt)
    ax.legend(handles=handles, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")
