"""vllm serve 生命周期：渲染启动脚本 / 启动 / 就绪检测 / 停止。

设计原则：env 全部由 bash 模板内 `export` 管理，Python 只负责拉起 bash。
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .deployments import Deployment


def render_start_script(dep: Deployment, run_dir: Path, template_path: Path,
                        model_path: str, port: int) -> Path:
    """读模板、替换占位符、写到 run_dir/start_vllm.sh，chmod +x。"""
    tpl = template_path.read_text(encoding="utf-8")
    substitutions = {
        "__DEPLOYMENT_NAME__": dep.name,
        "__LOG_DIR__": str(run_dir),
        "__MODEL_PATH__": model_path,
        "__PORT__": str(port),
        "__TP__": str(dep.tp),
        "__DP__": str(dep.dp),
        "__MAX_NUM_SEQS__": str(dep.max_num_seqs),
        # JSON list, no spaces — vllm expects a literal list in the compilation-config JSON
        "__CGCS__": json.dumps(dep.cudagraph_capture_sizes, separators=(",", "")),
    }
    text = tpl
    for k, v in substitutions.items():
        text = text.replace(k, v)
    if "__" in text:
        # 检测未替换的占位符
        leftovers = [tok for tok in text.split() if tok.startswith("__") and tok.endswith("__")]
        if leftovers:
            raise RuntimeError(f"模板有未替换的占位符：{leftovers[:5]}")

    out = run_dir / "start_vllm.sh"
    out.write_text(text, encoding="utf-8")
    out.chmod(0o755)
    return out


def start_vllm(start_script: Path, log_path: Path) -> subprocess.Popen:
    """用 login shell 启动 vllm serve；stdout/stderr 合并到 log_path。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    # bash -lc 触发 login shell，加载 .bash_profile / conda activate / CANN setup 等
    cmd = ["bash", "-lc", f"bash {start_script}"]
    print(f"[vllm] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # 单独的进程组，便于一并 kill
    )
    pid_file = start_script.parent / "vllm.pid"
    pid_file.write_text(str(proc.pid))
    print(f"[vllm] pid={proc.pid}, log={log_path}")
    return proc


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, socket.timeout, ConnectionResetError):
        return False
    except Exception:
        return False


def wait_until_ready(host: str, port: int, proc: subprocess.Popen,
                     timeout: float = 900.0, interval: float = 5.0) -> bool:
    """轮询 /v1/models 直到 200；进程提前死掉则立即返回 False。"""
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    print(f"[vllm] waiting for ready @ {url} (timeout={timeout:.0f}s)")
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[vllm] process exited early with code={proc.returncode}")
            return False
        if _http_ok(url):
            print(f"[vllm] READY after {timeout - (deadline - time.time()):.1f}s")
            return True
        time.sleep(interval)
    print(f"[vllm] TIMEOUT after {timeout:.0f}s")
    return False


def stop_vllm(proc: subprocess.Popen, grace_s: float = 60.0) -> None:
    """SIGTERM 整个进程组 → 等 grace → 仍存活则 SIGKILL。"""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    print(f"[vllm] SIGTERM pgid={pgid}")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
        print(f"[vllm] stopped gracefully")
        return
    except subprocess.TimeoutExpired:
        pass
    print(f"[vllm] grace timeout, SIGKILL pgid={pgid}")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("[vllm] WARN: proc still alive after SIGKILL")


def wait_port_free(host: str, port: int, timeout: float = 60.0) -> bool:
    """等到 host:port 不再被监听，避免下一个部署 bind 冲突。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                # 连得上说明还有人在监听
            except (ConnectionRefusedError, socket.timeout, OSError):
                return True
        time.sleep(1.0)
    return False
