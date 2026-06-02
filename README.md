# qwen_perf_test

910B3 Qwen3.6-35B-A3B-w8a8 vllm 性能 sweep 工具。一条命令跑完：

- **6 套部署**（tp/dp 组合）的自动启停
- 每套部署内遍历各自的 batch_size 序列
- 每个 batch_size 跑 ar/en/es/pt/zh **5 个语言数据集**
- 每次测试**后台采样** CPU% / NPU AI core% / NPU 显存（`npu-smi info`）
- 输出：汇总 CSV、单卡平均吞吐 CSV、单部署双面板图、6 部署聚合图、**单卡吞吐对比图**
- E2EL **两档门槛**：60s（橙）/ 120s（红），分别在图上以不同样式标记

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
├── merge_runs.py                # 合并多次同配置 sweep 的结果（CLI 入口）
├── perf_sweep/
│   ├── deployments.py           # 6 个 Deployment + 全局路径/端口/数据集
│   ├── vllm_server.py           # 渲染脚本 / 启停 vllm / ready check
│   ├── ais_bench_runner.py      # patch_config + run_one（调用 ais_bench）
│   ├── result_parser.py         # 解析 ais_bench 的 CSV/JSON
│   ├── resource_monitor.py      # CPU/内存/NPU AI core+显存 后台采样
│   ├── plotter.py               # 单部署双面板图 + 聚合图 + 单卡吞吐对比图
│   ├── merge.py                 # 合并多份 perf_summary.csv → 一份完整结果
│   └── orchestrator.py          # 主循环 + 增量 CSV + 单卡汇总
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
│   ├── vllm.pid
│   └── resource_samples/        # 仅 --monitor-raw 时；按 bs/lang 一份原始时序 TSV
├── tp2_dp2/  ...                # 其余 5 个部署同结构
├── perf_summary.csv             # 全部 6 部署聚合，含 CPU/NPU 资源列
├── perf_summary_tp2_dp1.csv     # 6 个分部署 CSV
├── per_card_summary.csv         # (deployment, batch_size) 跨数据集均值 / 单卡吞吐
├── perf_throughput.png          # 2×3 聚合图（throughput 面板，橙环=E2EL>60s, 红实心=>120s）
├── per_card_throughput.png      # 单卡吞吐对比图（每部署一条曲线）
└── perf_throughput_tp2_dp1.png  # 6 个单部署双面板图
```

---

## 环境前置

脚本本身需要在 **vllm 同一台远程 Linux** 上运行。前置条件：

- `vllm`、`ais_bench` 命令在 PATH 里
- 模型权重位于 `/data/q00931063/Qwen3.6-35B-A3B-w8a8/`（在 `perf_sweep/deployments.py` 改 `MODEL_PATH`）
- 数据集 5 个 jsonl 位于 `/data/q00931063/test/{ar,en,es,pt,zh}.jsonl`（同上文件改 `DATASETS`）
- `/workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py` 存在（脚本会原地改 `batch_size`/`max_out_len` 两个字段）
- `npu-smi` 在 PATH 里（资源监控的 NPU 数据来源；缺失时仅 NPU 字段留空，CPU/内存仍能采）
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
E2EL_BUDGETS     = [60.0, 120.0]   # 警告 / 严重 两档；越严重越靠后
RESOURCE_SAMPLE_INTERVAL_S = 2.0
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
| `--no-monitor`           | 关闭资源监控（CPU/NPU/显存采样） |
| `--monitor-interval 2.0` | 资源采样间隔秒数（默认 2.0s） |
| `--monitor-raw`          | 额外把每次采样原始值落到 `<dep>/resource_samples/bs{N}_{lang}.tsv` 便于事后画时序图 |

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

**5. 检查聚合图与单卡吞吐**

- `runs/<时间戳>/perf_throughput.png` —— 2×3 子图，每格一个部署，橙环=E2EL>60s，红实心=E2EL>120s
- `runs/<时间戳>/per_card_throughput.png` —— 6 条曲线一张图，**横向对比哪种部署单卡最划算**
- `runs/<时间戳>/per_card_summary.csv` —— 每行 = 一个 (deployment, batch_size) 的均值 + 单卡均吞吐

---

## 合并多次结果（merge_runs.py）

同一组 tp/dp 部署分多次跑完时（典型是每次只覆盖不同的 `batch_sizes` /
`cudagraph_capture_sizes`），用 `merge_runs.py` 把多份 `perf_summary.csv` 拼成一份
完整实验结果，并重算单卡汇总、重画全套图：

```bash
# 两次跑合成一份（不指定 --out 时落到 runs/merged_<时间戳>/）
python3 merge_runs.py runs/20260601_090000 runs/20260601_140000

# 指定输出目录
python3 merge_runs.py runs/a runs/b --out runs/merged_full

# 也可直接指向 csv
python3 merge_runs.py runs/a/perf_summary.csv runs/b/perf_summary.csv
```

合并语义：

- 以 `(tp, dp, batch_size, dataset)` 为唯一键。
- 按命令行给出的**输入顺序，靠后的 run 覆盖靠前的同键行**，每条覆盖都打印
  `[warn]` 告警。**把你认为更准的那次放在后面。**
- 产物与单次 sweep 完全对齐：`perf_summary.csv`、各 `perf_summary_<dep>.csv`、
  `per_card_summary.csv`，以及 `perf_throughput_<dep>.png` /
  `perf_throughput.png`（多部署时）/ `per_card_throughput.png`。

两点约束（设计使然）：

1. `cudagraph_capture_sizes` 不进 CSV，合并结果也不体现它——它只影响数值本身。
   重建出的 Deployment 仅用 tp/dp + 实际出现过的 batch_size 驱动聚合/画图。
2. 合并只读历史 CSV，**不重跑 ais_bench**；分部署 `vllm.log` / `resource_samples/`
   等原始产物仍各留在原 run 目录里，不会被搬进合并目录。

---

## 输出说明

### `perf_summary.csv` / `perf_summary_<dep>.csv`

每行 = 一次 `(deployment, batch_size, dataset)` 的测试。字段：

| 列 | 说明 |
|---|---|
| `deployment` | `tp{tp}_dp{dp}` |
| `tp` / `dp` | 数字 |
| `batch_size` | ais_bench 并发数 |
| `dataset` | 语言代码 (ar/en/es/pt/zh) |
| `num_prompt` | = batch_size × 2 |
| `E2EL(ms)` / `TTFT(ms)` / `TPOT(ms)` / `ITL(ms)` | ais_bench CSV 的 Average 列 |
| `OTT(token/s)` | Output Token Throughput（ais_bench JSON） |
| `cpu_util_avg(%)` / `cpu_util_max(%)` | 测试期间 CPU 利用率（/proc/stat 差分） |
| `mem_util_avg(%)` / `mem_util_max(%)` | 系统内存使用率（/proc/meminfo） |
| `npu_aicore_avg(%)` / `npu_aicore_max(%)` | "在用"卡（任一快照显存 ≥ 1GB）的 AI core 利用率 |
| `npu_mem_used_max(MB)` | 单卡显存峰值（MB） |
| `npu_mem_used_max_chip` | 峰值发生在哪张卡（chip id） |
| `npu_loaded_chips` | 该次测试期间被认定"在用"的 chip 列表（逗号分隔） |
| `resource_samples` | 实际采到的样本数（间隔 2s × run 时长） |
| `source_file` | 解析时取到的 ais_bench csv 路径，便于回溯 |

非 Ascend 机器或 `npu-smi` 不可用时，`npu_*` 列留空；`--no-monitor` 时所有资源列留空。

### `per_card_summary.csv`

每行 = 一个 `(deployment, batch_size)` 跨 5 个数据集的均值。**这是横向对比哪种部署单卡最划算的核心表**。

| 列 | 说明 |
|---|---|
| `deployment` / `tp` / `dp` / `n_cards` | `n_cards = tp × dp` |
| `batch_size` |  |
| `n_datasets` | 实际参与均值的数据集数（默认 5） |
| `OTT_mean(token/s)` | 跨数据集 OTT 均值（整机吞吐） |
| `OTT_per_card_mean(token/s)` | = `OTT_mean / n_cards` |
| `E2EL_mean(ms)` / `E2EL_max(ms)` | 跨数据集 E2EL 均值/最大值 |
| `worst_E2EL_tier` | 该组合跨数据集见过的最高门槛档（-1 = 都没超过, 0 = >60s, 1 = >120s） |

### 图

- **单部署图** `perf_throughput_<dep>.png`：双面板 —— 上 throughput vs concurrency（橙环=E2EL>60s, 红实心=>120s）；下 E2EL vs concurrency（两条虚线分别是 60s/120s 门槛）
- **聚合图** `perf_throughput.png`：2×3 子图，每格一个部署的 throughput 面板，shared y 轴便于横向对比
- **单卡吞吐对比图** `per_card_throughput.png`：6 条曲线一张图，横轴 batch_size，纵轴单卡吞吐（跨 5 数据集均值），同样的 E2EL 视觉编码

---

## 常见问题

**Q: 脚本启动 vllm 跟我手动启动的环境变量一致吗？**

A: 设计上一致 —— 9 个 export 都在 `templates/start_vllm.sh.tpl` 里原样保留；`bash -lc` 加载 `.bashrc` / conda activate / CANN setup 等隐式 env。可用上面"验证步骤 2" `diff vllm_env.log` 实测，且你随时可以 `bash runs/.../start_vllm.sh` 手动复现。

**Q: ready check 一直 timeout 怎么办？**

A: 看 `runs/<时间戳>/<dep>/vllm.log` —— 通常是 cudagraph capture 慢、显存不够、或某个 env 漏了。`--ready-timeout 1800` 给到 30 分钟，仍 timeout 就说明 vllm 自身报错。`vllm_env.log` 跟手动 env 做 diff 是排查 env 缺失最快的方法。

**Q: 想中途补跑某个部署？**

A: `python3 run_perf_sweep.py --only tp4_dp2 --run-root runs/<已存在的目录>` —— 注意这会**覆盖** `perf_summary_tp4_dp2.csv` 与聚合 CSV，其它部署的分部署 CSV 保留。

**Q: 同一组 tp/dp 我分了好几次跑（每次不同 batch_sizes / capture_sizes），怎么拼成一份完整结果？**

A: 用 `merge_runs.py`，见上文「合并多次结果」。各次跑各自落在独立的 `runs/<时间戳>/`，跑完再 `python3 merge_runs.py runs/第一次 runs/第二次` 合并即可——靠后的输入覆盖靠前的同键行，不要用 `--run-root` 指向同一目录互相覆盖。

**Q: ais_bench 报错怎么办？**

A: 脚本捕获 stderr 末尾 2000 字符打到 stdout。也可以去 ais_bench `--work-dir`（默认 `/data/q00931063/910B3_result`）下找对应时间戳目录。

**Q: 想跳过 vllm 启停，自己手动起服务调试 sweep 链路？**

A: 手动启动 vllm 后：`python3 run_perf_sweep.py --skip-launch --only tp4_dp2 --batch-sizes 100 --datasets en`。

**Q: 资源监控开销大吗？怎么看到原始时序？**

A: 默认 2s 采一次，调一次 `npu-smi info` 大约几十 ms；对 ais_bench perf 测试的影响可以忽略。要看原始时序加 `--monitor-raw`，每次 run 会写一个 `<dep>/resource_samples/bs{N}_{lang}.tsv`，每行 `t_relative cpu mem npu_chips_list`，可以直接 `awk` / `gnuplot` 画时序图。

**Q: 怎么改 E2EL 门槛？比如想加一档 30s？**

A: 改 `perf_sweep/deployments.py` 的 `E2EL_BUDGETS`，从严到宽顺序排即可（如 `[30.0, 60.0, 120.0]`）。`plotter.py` 的 `_TIER_STYLES` 也要补一档样式（再加一个 dict），否则会复用最后一档样式。
