"""6 套 vllm 部署配置 + 全局常量。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Deployment:
    tp: int
    dp: int
    batch_sizes: list[int]
    cudagraph_capture_sizes: list[int]

    @property
    def name(self) -> str:
        return f"tp{self.tp}_dp{self.dp}"

    @property
    def max_num_seqs(self) -> int:
        return max(self.batch_sizes)


_SMALL_BS = [50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
_SMALL_CGCS = [200, 240, 280, 320, 360, 400, 800, 1200, 1600, 2000]
_LARGE_BS = [150, 200, 300, 400, 500, 600, 700, 800, 900]
_LARGE_CGCS = [600, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600]

DEPLOYMENTS: list[Deployment] = [
    Deployment(tp=2, dp=1, batch_sizes=_SMALL_BS, cudagraph_capture_sizes=_SMALL_CGCS),
    Deployment(tp=2, dp=2, batch_sizes=_SMALL_BS, cudagraph_capture_sizes=_SMALL_CGCS),
    Deployment(tp=4, dp=1, batch_sizes=_SMALL_BS, cudagraph_capture_sizes=_SMALL_CGCS),
    Deployment(tp=8, dp=1, batch_sizes=_LARGE_BS, cudagraph_capture_sizes=_LARGE_CGCS),
    Deployment(tp=4, dp=2, batch_sizes=_LARGE_BS, cudagraph_capture_sizes=_LARGE_CGCS),
    Deployment(tp=2, dp=4, batch_sizes=_LARGE_BS, cudagraph_capture_sizes=_LARGE_CGCS),
]

# ---------------------- 路径 / 端口 ----------------------
MODEL_PATH = "/data/q00931063/Qwen3.6-35B-A3B-w8a8/"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 1025

AIS_CONFIG_PATH = "/workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
AIS_WORK_DIR = "/data/q00931063/910B3_result"

# ---------------------- 数据集 ----------------------
DATASETS: dict[str, str] = {
    "ar": "/data/q00931063/test/ar.jsonl",
    "en": "/data/q00931063/test/en.jsonl",
    "es": "/data/q00931063/test/es.jsonl",
    "pt": "/data/q00931063/test/pt.jsonl",
    "zh": "/data/q00931063/test/zh.jsonl",
}

# ---------------------- 评测参数 ----------------------
MAX_OUT_LEN = 200

# E2E 延迟门槛（秒），从严到宽。画图时按列表顺序叠加视觉标记：
#   E2EL_BUDGETS[0]  = 警告  (橙)
#   E2EL_BUDGETS[1]  = 严重  (红)
E2EL_BUDGETS: list[float] = [60.0, 120.0]
# 资源监控采样间隔（秒）。非 Ascend 机器仍会采 CPU/内存。
RESOURCE_SAMPLE_INTERVAL_S = 2.0

# ---------------------- 项目内相对路径 ----------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "start_vllm.sh.tpl"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs"
