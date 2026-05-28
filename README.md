# qwen_perf_test

910B3 Qwen3.6-35B-A3B-w8a8 vllm 性能 sweep 工具。一条命令跑完：

- **6 套部署**（tp/dp 组合）的自动启停
- 每套部署内遍历各自的 batch_size 序列
- 每个 batch_size 跑 ar/en/es/pt/zh **5 个语言数据集**
- 汇总 CSV + 单部署双面板图 + 6 部署聚合图

部署/参数矩阵：

| TP | DP | batch_sizes                              | cudagraph_capture_sizes                                |
|----|----|------------------------------------------|--------------------------------------------------------|
| 2  | 1  | 50,60,70,80,90,100,200,300,400,500       | 200,240,280,320,360,400,800,1200,1600,2000             |
| 2  | 2  | 50,60,70,80,90,100,200,300,400,500       | 200,240,280,320,360,400,800,1200,1600,2000             |
| 4  | 1  | 50,60,70,80,90,100,200,300,400,500       | 200,240,280,320,360,400,800,1200,1600,2000             |
| 8  | 1  | 150,200,300,400,500,600,700,800,900      | 600,800,1200,1600,2000,2400,2800,3200,3600             |
| 4  | 2  | 150,200,300,400,500,600,700,800,900      | 600,800,1200,1600,2000,2400,2800,3200,3600             |
| 2  | 4  | 150,200,300,400,500,600,700,800,900      | 600,800,1200,1600,2000,2400,2800,3200,3600             |

`--max-num-seqs` 自动取该部署 `max(batch_sizes)`，其它 vllm serve 参数沿用手动启动的命令。

---

## 目录结构

```
qwen_perf_test/
├── run_perf_sweep.py            # CLI 入口
├── perf_sweep/
│   ├── deployments.py           # 6 个 Deployment + 全局路径/端口/数据集
│   ├── vllm_server.py           # 渲染脚本 / 启停 vllm / ready check
│   ├── ais_bench_runner.py      # patch_config + run_one（调用 ais_bench）
│   ├── result_parser.py         # 解析 ais_bench 的 CSV/JSON
│   ├── plotter.py               # 单部署双面板图 + 6 聚合子图
│   └── orchestrator.py          # 主循环 + 增量 CSV
└── templates/
    └── start_vllm.sh.tpl        # vllm 启动 bash 模板（env 全在 bash 里 export）
```

运行时输出（默认在 `./runs/<时间戳>/`）：

```
runs/20260528_153000/
├── tp2_dp1/
│   ├── start_vllm.sh            # 渲染好的实际启动脚本（cat 它就能手动复现）
│   ├── vllm.log                 # vllm serve 的 stdout+stderr
│   ├── vllm_env.log             # 启动瞬间 env|sort 的快照（用于对比手动启动）
│   ├── vllm_which.log           # which vllm + vllm --version
│   └── vllm.pid
├── tp2_dp2/  ...                # 其余 5 个部署同结构
├── perf_summary.csv             # 全部 6 部署聚合，有 deployment/tp/dp 列
├── perf_summary_tp2_dp1.csv     # 6 个分部署 CSV
├── perf_throughput.png          # 2×3 聚合图（throughput 面板，红圈=E2EL>60s）
└── perf_throughput_tp2_dp1.png  # 6 个单部署双面板图
```

---

## 环境前置

脚本本身需要在 **vllm 同一台远程 Linux** 上运行。前置条件：

- `vllm`、`ais_bench` 命令在 PATH 里
- 模型权重位于 `/data/q00931063/Qwen3.6-35B-A3B-w8a8/`（在 `perf_sweep/deployments.py` 改 `MODEL_PATH`）
- 数据集 5 个 jsonl 位于 `/data/q00931063/test/{ar,en,es,pt,zh}.jsonl`（同上文件改 `DATASETS`）
- `/workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py` 存在（脚本会原地改 `batch_size`/`max_out_len` 两个字段）
- `matplotlib`（可选 —— 没装会跳过画图，CSV 仍正常生成）

NPU/CANN 相关 env 不需要在 Python 里管理，由 `templates/start_vllm.sh.tpl` 里的 `export` 和 `bash -lc` 加载用户 profile 共同负责。

---

## 配置修改

要改实验范围、路径、端口、模型，**只改一个文件**：`perf_sweep/deployments.py`

```python
MODEL_PATH       = "/data/q00931063/Qwen3.6-35B-A3B-w8a8/"
VLLM_HOST        = "127.0.0.1"
VLLM_PORT        = 1025
AIS_CONFIG_PATH  = "/workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
AIS_WORK_DIR     = "/data/q00931063/910B3_result"
DATASETS         = {...}
MAX_OUT_LEN      = 200
E2EL_BUDGET_S    = 60.0
DEPLOYMENTS      = [Deployment(tp=2, dp=1, batch_sizes=[...], cudagraph_capture_sizes=[...]), ...]
```

要改 vllm serve 的其它启动参数（`--seed`、`--max-model-len`、`--gpu-memory-utilization` 等）：改 `templates/start_vllm.sh.tpl`，**和你手动启动用的命令保持字面一致**即可。

---

## 使用方法

最简单：

```bash
python3 run_perf_sweep.py
```

这会按 `DEPLOYMENTS` 顺序，对 6 个部署各启停一次 vllm，跑完后把 CSV 和图写到 `./runs/<时间戳>/`。

### CLI 选项

| 选项 | 用途 |
|---|---|
| `--only tp4_dp2,tp8_dp1` | 只跑指定部署 |
| `--datasets en,zh`       | 只跑指定语言数据集 |
| `--batch-sizes 50,100`   | 覆盖各部署默认 batch_sizes（调试用） |
| `--run-root <dir>`       | 指定输出根目录（默认 `./runs/<时间戳>/`） |
| `--ready-timeout 900`    | vllm 起来后等 `/v1/models` 的超时秒数（默认 900s，首次编译/cudagraph 慢） |
| `--max-out-len 200`      | 覆盖默认输出长度 |
| `--config <path>`        | 覆盖 ais_bench 的 `vllm_api_stream_chat.py` 路径 |
| `--work-dir <path>`      | 覆盖 ais_bench `--work-dir` |
| `--dry-run`              | 只渲染 6 份 `start_vllm.sh`，不真启 vllm，也不跑 ais_bench |
| `--skip-launch`          | 不启动 vllm（你已手动起好），仅跑 sweep |

### 上线前推荐验证顺序

**1. 检查渲染出的启动脚本**

```bash
python3 run_perf_sweep.py --only tp2_dp1 --dry-run
cat runs/<时间戳>/tp2_dp1/start_vllm.sh
```

跟你手动启动的命令逐行对照。

**2. env 一致性验证**（最关键的一步）

```bash
python3 run_perf_sweep.py --only tp2_dp1 --batch-sizes 50 --datasets en
```

vllm 起来后另开终端 ssh 到同机：

```bash
env | sort > /tmp/manual_env.log
diff /tmp/manual_env.log runs/<时间戳>/tp2_dp1/vllm_env.log
```

应该只差 `_` / `SHLVL` / `PWD` / 终端相关的几行。如果有 `LD_LIBRARY_PATH` / `ASCEND_*` 这种实质性变量缺失，说明 `bash -lc` 没加载到你的 profile，需要补 `.bashrc` / `~/.bash_profile`。

**3. 单部署完整跑通**

```bash
python3 run_perf_sweep.py --only tp2_dp1
```

预计 10 batch_size × 5 lang ≈ 50 次 run。检查：

- `runs/<时间戳>/perf_summary_tp2_dp1.csv` 行数 = 50
- `runs/<时间戳>/perf_throughput_tp2_dp1.png` 双面板图正常

**4. 全量 sweep**

```bash
nohup python3 run_perf_sweep.py > sweep.out 2>&1 &
tail -f sweep.out
```

中途允许 Ctrl-C —— 由于增量写 CSV，已完成的行不会丢；`vllm` 子进程组会被 `SIGTERM` 干净杀掉，验证：`ps -ef | grep vllm` 应为空。

**5. 检查聚合图**

`runs/<时间戳>/perf_throughput.png` —— 2×3 子图，每格一个部署，红圈标 E2EL>60s。

---

## 输出说明

### `perf_summary.csv` / `perf_summary_<dep>.csv`

字段：

| 列 | 说明 |
|---|---|
| `deployment` | `tp{tp}_dp{dp}` |
| `tp` / `dp` | 数字 |
| `batch_size` | ais_bench 并发数 |
| `dataset` | 语言代码 (ar/en/es/pt/zh) |
| `num_prompt` | = batch_size × 2 |
| `E2EL(ms)` / `TTFT(ms)` / `TPOT(ms)` / `ITL(ms)` | ais_bench CSV 的 Average 列 |
| `OTT(token/s)` | Output Token Throughput（ais_bench JSON） |
| `source_file` | 解析时取到的 ais_bench csv 路径，便于回溯 |

### 图

- **单部署图** `perf_throughput_<dep>.png`：双面板 —— 上 throughput vs concurrency（红圈=E2EL>60s）；下 E2EL vs concurrency（红虚线=60s 预算）
- **聚合图** `perf_throughput.png`：2×3 子图，每格一个部署的 throughput 面板，shared y 轴便于横向对比

---

## 常见问题

**Q: 脚本启动 vllm 跟我手动启动的环境变量一致吗？**

A: 设计上一致 —— 9 个 export 都在 `templates/start_vllm.sh.tpl` 里原样保留；`bash -lc` 加载 `.bashrc` / conda activate / CANN setup 等隐式 env。可用上面"验证步骤 2" `diff vllm_env.log` 实测，且你随时可以 `bash runs/.../start_vllm.sh` 手动复现。

**Q: ready check 一直 timeout 怎么办？**

A: 看 `runs/<时间戳>/<dep>/vllm.log` —— 通常是 cudagraph capture 慢、显存不够、或某个 env 漏了。`--ready-timeout 1800` 给到 30 分钟，仍 timeout 就说明 vllm 自身报错。`vllm_env.log` 跟手动 env 做 diff 是排查 env 缺失最快的方法。

**Q: 想中途补跑某个部署？**

A: `python3 run_perf_sweep.py --only tp4_dp2 --run-root runs/<已存在的目录>` —— 注意这会**覆盖** `perf_summary_tp4_dp2.csv` 与聚合 CSV，其它部署的分部署 CSV 保留。

**Q: ais_bench 报错怎么办？**

A: 脚本捕获 stderr 末尾 2000 字符打到 stdout。也可以去 ais_bench `--work-dir`（默认 `/data/q00931063/910B3_result`）下找对应时间戳目录。

**Q: 想跳过 vllm 启停，自己手动起服务调试 sweep 链路？**

A: 手动启动 vllm 后：`python3 run_perf_sweep.py --skip-launch --only tp4_dp2 --batch-sizes 100 --datasets en`。
