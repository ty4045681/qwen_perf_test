"""画图：单部署双面板 + 6 部署聚合。"""
from __future__ import annotations

from pathlib import Path

from .deployments import Deployment, E2EL_BUDGET_S


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


def _draw_throughput(ax, by_lang: dict, lang_color: dict,
                     *, with_legend: bool, title: str | None) -> None:
    """画 Throughput vs Concurrency 曲线 + E2EL 超预算红圈。"""
    for lang, lst in sorted(by_lang.items()):
        xs = [r["batch_size"] for r in lst]
        ys = [float(r["OTT(token/s)"]) for r in lst]
        ax.plot(xs, ys, marker="o", color=lang_color[lang],
                label=lang, linewidth=1.8)
        for r in lst:
            try:
                e2el_s = float(r["E2EL(ms)"]) / 1000.0
            except (TypeError, ValueError):
                continue
            if e2el_s > E2EL_BUDGET_S:
                ax.scatter([r["batch_size"]], [float(r["OTT(token/s)"])],
                           s=140, facecolors="none", edgecolors="red",
                           linewidths=1.8, zorder=5)
    ax.set_ylabel("Output Throughput (token/s)")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if with_legend and by_lang:
        ax.legend(title="dataset", loc="upper left", ncol=max(1, len(by_lang)))


def _draw_e2el(ax, by_lang: dict, lang_color: dict, all_bs: list[int]) -> None:
    """下面板：E2EL vs Concurrency + 预算虚线。"""
    for lang, lst in sorted(by_lang.items()):
        xs, ys = [], []
        for r in lst:
            try:
                e2el_s = float(r["E2EL(ms)"]) / 1000.0
            except (TypeError, ValueError):
                continue
            xs.append(r["batch_size"])
            ys.append(e2el_s)
        if xs:
            ax.plot(xs, ys, marker="o", color=lang_color[lang],
                    linewidth=1.8, label=lang)
    ax.axhline(E2EL_BUDGET_S, color="red", linestyle="--", linewidth=1.2,
               label=f"E2EL budget = {E2EL_BUDGET_S:.0f}s")
    ax.set_xlabel("Concurrency (batch_size)")
    ax.set_ylabel("E2E Latency (s)")
    if all_bs:
        ax.set_xticks(all_bs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=max(1, len(by_lang) + 1), fontsize=8)


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

    _draw_throughput(
        ax_top, by_lang, lang_color,
        with_legend=True,
        title=(f"910B3 Qwen3.6-35B-A3B-w8a8 [{dep.name}]: "
               f"Throughput & E2E Latency vs Concurrency\n"
               f"(red open circle = E2EL > {E2EL_BUDGET_S:.0f}s)"),
    )
    all_bs = sorted({r["batch_size"] for lst in by_lang.values() for r in lst})
    _draw_e2el(ax_bot, by_lang, lang_color, all_bs)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


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

    # 收集全局 lang 集合，保证子图颜色一致
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
        _draw_throughput(
            ax, by_lang, lang_color,
            with_legend=False,
            title=dep.name,
        )
        ax.set_xlabel("Concurrency (batch_size)")

    # 多出来的格子隐藏
    for ax in axes[len(deployments):]:
        ax.set_visible(False)

    # 统一 legend 放在 figure 顶部
    handles = [plt.Line2D([0], [0], marker="o", color=lang_color[lang],
                          label=lang, linewidth=1.8)
               for lang in all_langs]
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor="none", markeredgecolor="red",
                              markersize=10, markeredgewidth=1.8,
                              label=f"E2EL > {E2EL_BUDGET_S:.0f}s"))
    fig.legend(handles=handles, loc="upper center",
               ncol=len(all_langs) + 1, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("910B3 Qwen3.6-35B-A3B-w8a8: Throughput vs Concurrency across deployments",
                 y=1.06, fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")
