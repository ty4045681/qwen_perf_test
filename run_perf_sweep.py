#!/usr/bin/env python3
"""910B3 Qwen3.6-35B-A3B-w8a8 性能 sweep — CLI 入口。

遍历 6 个 vllm 部署 × 各部署对应的 batch_size × 5 个语言数据集。
每个部署会自动启停 vllm serve；详情见 perf_sweep/orchestrator.py。
"""
import argparse
import os
import sys

from perf_sweep.deployments import (
    AIS_CONFIG_PATH,
    AIS_WORK_DIR,
    MAX_OUT_LEN,
    RESOURCE_SAMPLE_INTERVAL_S,
    VLLM_HOST,
)
from perf_sweep.orchestrator import run


def _ensure_no_proxy_for_localhost() -> None:
    """把本机回环地址加进 no_proxy/NO_PROXY。

    vllm 跑在 VLLM_HOST 上，就绪检测用的 urllib（vllm_server._http_ok）和
    ais_bench 压测子进程都直连这个地址。若环境里设了 http(s)_proxy 而 no_proxy
    未包含 127.0.0.1，请求会被丢给代理 → 卡在 waiting for ready / 压测连不上
    （此时 curl 反而能通，因为 curl 默认绕过 localhost）。这里在进程内补齐，
    确保 urllib 与继承本进程 env 的子进程都直连。
    """
    local = ["127.0.0.1", "localhost", "::1", VLLM_HOST]
    for var in ("no_proxy", "NO_PROXY"):
        parts = [p.strip() for p in os.environ.get(var, "").split(",") if p.strip()]
        for h in local:
            if h not in parts:
                parts.append(h)
        os.environ[var] = ",".join(parts)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=AIS_CONFIG_PATH,
                    help="ais_bench 的 vllm_api_stream_chat.py 路径")
    ap.add_argument("--work-dir", default=AIS_WORK_DIR,
                    help="ais_bench --work-dir，每次 perf 会在这下面产时间戳目录")
    ap.add_argument("--run-root", default=None,
                    help="本次 sweep 的输出根目录（默认 ./runs/<timestamp>/）")
    ap.add_argument("--max-out-len", type=int, default=MAX_OUT_LEN)

    ap.add_argument("--only", default="",
                    help="只跑指定部署，逗号分隔（如 tp4_dp2,tp8_dp1）")
    ap.add_argument("--datasets", default="",
                    help="只跑指定数据集，逗号分隔（如 en,zh）")
    ap.add_argument("--batch-sizes", default="",
                    help="覆盖各部署默认 batch_sizes，逗号分隔（调试用）")

    ap.add_argument("--ready-timeout", type=float, default=900.0,
                    help="vllm serve 启动后等待 /v1/models 200 的超时（秒）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只渲染 start_vllm.sh，不真启动 vllm 也不跑 ais_bench")
    ap.add_argument("--skip-launch", action="store_true",
                    help="不启动 vllm（假定已手动启动好），仅跑 sweep")

    ap.add_argument("--no-monitor", action="store_true",
                    help="关闭资源监控（CPU/NPU AI core/显存）")
    ap.add_argument("--monitor-interval", type=float,
                    default=RESOURCE_SAMPLE_INTERVAL_S,
                    help="资源采样间隔秒数（默认 2.0s）")
    ap.add_argument("--monitor-raw", action="store_true",
                    help="每次 run_one 把每次采样的原始值落到 "
                         "<dep>/resource_samples/bs{N}_{lang}.tsv，便于事后画时序图")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    _ensure_no_proxy_for_localhost()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
