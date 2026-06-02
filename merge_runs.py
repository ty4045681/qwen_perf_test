#!/usr/bin/env python3
"""合并多次同配置 sweep 的结果 — CLI 入口。

把分多次 run_perf_sweep.py 产出的 perf_summary.csv 拼成一份完整实验结果：
合并总表、分部署表、单卡吞吐汇总，以及全套图。

以 (tp, dp, batch_size, dataset) 为唯一键；按命令行给出的输入顺序，后面的
run 覆盖前面的同键行（每条覆盖都会打印告警）。详见 perf_sweep/merge.py。

示例：
    # 两次跑（不同 batch_sizes / capture_sizes）合成一份
    python merge_runs.py runs/20260601_090000 runs/20260601_140000

    # 指定输出目录
    python merge_runs.py runs/a runs/b --out runs/merged_full

    # 也可直接指向 csv
    python merge_runs.py runs/a/perf_summary.csv runs/b/perf_summary.csv
"""
import argparse
import sys

from perf_sweep.merge import run_merge


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="run 根目录（含 perf_summary.csv）或直接的 perf_summary.csv，"
                         "按顺序排列：靠后的覆盖靠前的同键行")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 ./runs/merged_<timestamp>/）")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_merge(args.inputs, args.out)


if __name__ == "__main__":
    sys.exit(main())
