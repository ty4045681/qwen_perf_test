"""ais_bench 调用相关：原地改 vllm_api_stream_chat.py + 跑 perf。"""
import re
import subprocess
import time
from pathlib import Path

from .deployments import AIS_CONFIG_PATH


def patch_config(batch_size: int, max_out_len: int,
                 config_path: str = AIS_CONFIG_PATH) -> None:
    """原地修改 ais_bench 的 vllm_api_stream_chat.py，更新 batch_size 与 max_out_len。"""
    p = Path(config_path)
    text = p.read_text(encoding="utf-8")

    new_text, n1 = re.subn(r"(batch_size\s*=\s*)\d+", rf"\g<1>{batch_size}", text, count=1)
    new_text, n2 = re.subn(r"(max_out_len\s*=\s*)\d+", rf"\g<1>{max_out_len}", new_text, count=1)
    if n1 == 0 or n2 == 0:
        raise RuntimeError(f"未能在 {config_path} 中找到 batch_size / max_out_len 字段")
    p.write_text(new_text, encoding="utf-8")
    print(f"[patch] batch_size={batch_size}, max_out_len={max_out_len}")


def run_one(batch_size: int, dataset_path: str, work_dir: str) -> None:
    """运行一次 ais_bench perf 测试。"""
    num_prompt = batch_size * 2
    cmd = [
        "ais_bench",
        "--mode", "perf",
        "--models", "vllm_api_stream_chat",
        "--custom-dataset-data-type", "qa",
        "--custom-dataset-path", dataset_path,
        "--num-prompt", str(num_prompt),
        "--work-dir", work_dir,
    ]
    print(f"[run ] bs={batch_size} dataset={Path(dataset_path).stem} num_prompt={num_prompt}")
    print("       " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"       FAILED in {dt:.1f}s (code={proc.returncode})")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
    else:
        print(f"       OK in {dt:.1f}s")
