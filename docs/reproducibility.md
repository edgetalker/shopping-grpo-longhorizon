# 复现实验说明

本文档记录最终实验的输入、版本、命令、checkpoint 选择规则、导出流程和证据边界。
当前仓库处于包装交付阶段；下列训练、模型合并与 Final-200 命令用于复现说明，除非明确
决定重新执行，否则只运行文末的轻量验证。

## 1. 固定契约

| 项目 | 值 |
|---|---|
| 工作流 | Baseline → SFT → GRPO → Evaluation |
| Environment | ShopSimulator Environment v2.1 |
| Reward | Reward v3 |
| Observation / tool schema | v2 / v2 |
| Base model | Qwen3.5-2B |
| Python | 3.12（训练）；ShopSimulator 隔离环境为 3.10 |
| veRL / vLLM | 0.8.0 / 0.25.1 |
| PyTorch / Ray | 2.11.0 / 2.56.1 |
| Transformers | commit `7ea2320c76117e6742364808a666ef6f2fb40a67` |
| 训练硬件 | 单卡 96 GiB GPU |

依赖的权威声明在 `pyproject.toml` 和锁文件中。安装入口：

```bash
bash scripts/setup.sh
```

## 2. 数据清单

| 数据 | 数量 | SHA-256 |
|---|---:|---|
| GRPO train parquet | 1,000 | `98bb1799a42916b6ca2a1a2618f11cfec60f428e99e6ada448a78361768206e5` |
| GRPO validation parquet | 50 | `8c9fbdd22bf6d191d1da6bcca9c1612e170bb928483c4ce0c8d0934f9ba69317` |
| Final-200 tasks JSONL | 200 | `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5` |

本地核验：

```bash
shasum -a 256 \
  data/grpo/train.parquet \
  data/grpo/validation.parquet \
  data/evaluation/tasks.jsonl
```

Final-200 与 SFT、GRPO train/validation 的 task ID 零重叠。PureV4 SFT 与 GRPO
train 有 6 个历史 task-ID 重叠：`7682, 8993, 11298, 12380, 14953, 17219`。
两条 GRPO 臂使用同一冻结训练集，因此不影响两臂公平性，但讨论数据来源时必须披露。

## 3. 启动环境

```bash
bash scripts/start_environment.sh
```

默认地址为 `http://127.0.0.1:5700`。训练前应先执行运行时预检；`scripts/grpo.sh`
会自动调用 `scripts/check_grpo_runtime.py`。

## 4. SFT

记录的 SFT 配置：

- 数据：`data/sft_pure_v4/all.jsonl`，curriculum stage `c`；
- 实际 train/validation examples：1,069 / 118；
- 1 epoch，learning rate `5e-5`；
- LoRA rank/alpha `16/32`，all relevant linear targets；
- bf16、SDPA、gradient checkpointing；
- max length 24,576，seed 42。

复现命令：

```bash
BASE_MODEL=/path/to/Qwen3.5-2B

.venv/bin/python scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train data/sft_pure_v4/all.jsonl \
  --validation data/sft_pure_v4/all.jsonl \
  --curriculum-manifest data/sft_curriculum/manifest.json \
  --curriculum-stage c \
  --output outputs/models/sft-single-pass/adapter \
  --max-length 24576 \
  --epochs 1 \
  --learning-rate 5e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --dtype bf16 \
  --gradient-checkpointing \
  --attention-implementation sdpa

.venv/bin/python scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter outputs/models/sft-single-pass/adapter \
  --output outputs/models/sft-single-pass/merged \
  --bf16
```

记录结果：train loss `0.3983341980`，runtime 5,428 秒，peak GPU memory
68.97 GiB。关键哈希：

| 产物 | SHA-256 |
|---|---|
| SFT merged `model.safetensors` | `fe0a6e374cdbe1b401afc826254f830e96fabbff33cf84087d847760bb267c85` |
| SFT `adapter_model.safetensors` | `8885516ff96c9d91bd57f775e381157508d1fa4465a6c563e248031bac905aeb` |

## 5. Vanilla GRPO

主要配置位于 `configs/grpo.yaml`：

- four rollouts per prompt，temperature 0.7，top-p 0.9；
- prompt batch 2，PPO mini-batch 2，micro-batch 1；
- learning rate `1e-6`，3% warmup；
- LoRA rank/alpha `16/32`；
- `norm_adv_by_std_in_grpo=false`；
- reward KL、actor KL、entropy coefficient、length shaping 均关闭；
- dynamic sampling 最多 3 个 generation batches；
- 500 optimizer steps，每 50 步保存并验证。

先只解析命令：

```bash
bash scripts/grpo.sh \
  --model outputs/models/sft-single-pass/merged \
  --output outputs/models/grpo-vanilla-500-entropy-off-v1 \
  --experiment-name grpo-vanilla-500-entropy-off-v1 \
  --dry-run
```

实际执行时移除 `--dry-run`。不要在包装阶段执行。

## 6. Failure-only TRACE GRPO

除启用 `shopping_trace` 外，与 Vanilla 使用相同数据与优化配置：

```bash
SHOPPING_TRACE_FAILURE_ONLY=true \
bash scripts/grpo.sh \
  --model outputs/models/sft-single-pass/merged \
  --output outputs/models/grpo_trace_failure_only_v1 \
  --experiment-name grpo_trace_failure_only_v1 \
  --dry-run \
  -- shopping_trace.enable=true
```

实际执行时移除 `--dry-run`。TRACE 参数：

| 参数 | 值 |
|---|---:|
| gate | `exact_tie_failure_only` |
| epsilon | 0.1 |
| horizon | 3 |
| discount | 0.8 |
| terminal / outcome / turn weight | 2.0 / 1.0 / 0.2 |
| centered credit clip | `[-1, 1]` |
| scorer max sequence length | 24,576 |

## 7. Checkpoint 选择

Checkpoint 只能使用 50 题 validation mean reward 选择，Final-200 不参与选择。

| Step | Vanilla | TRACE |
|---:|---:|---:|
| 0 | 0.279744 | 0.254432 |
| 50 | 0.265869 | — |
| 100 | 0.285915 | 0.255382 |
| 150 | 0.255382 | 0.273148 |
| 200 | 0.280694 | 0.280694 |
| 250 | 0.224493 | 0.284639 |
| 300 | 0.273514 | 0.334576 |
| 350 | 0.310639 | 0.336038 |
| 400 | 0.299639 | 0.320014 |
| 450 | **0.327932** | **0.355292** |
| 500 | 0.316701 | 0.351023 |

两臂都选择 step 450。TRACE 的 50-step 值来自独立 smoke，不与后续 resumed full-run
曲线拼接成一个可比较 validation point。

## 8. LoRA 导出与正确合并

`scripts/export_grpo.sh` 只负责从 veRL checkpoint 恢复导出结构。学习到的 LoRA 可能
位于 `lora_adapter/`，不能直接假定 export 根目录就是已学习的独立模型。

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo-vanilla-500-entropy-off-v1/global_step_450/actor \
  outputs/models/grpo-vanilla-step450-merged

.venv/bin/python scripts/merge_lora_adapter.py \
  --base-model outputs/models/sft-single-pass/merged \
  --adapter outputs/models/grpo-vanilla-step450-merged/lora_adapter \
  --output outputs/models/grpo-vanilla-step450-deploy \
  --bf16
```

TRACE 同理，将输入与输出名替换为 `grpo_trace_failure_only_v1` 和
`grpo-trace-step450-*`。

部署前必须检查：

1. adapter 文件存在且非空；
2. deploy 目录的 `merge_manifest.json` 指向正确 base 和 adapter；
3. deploy 主权重不得与 SFT 主权重哈希相同；
4. 每个 vLLM 服务使用唯一 served name。

首次错误评测中，SFT 与两个错误 export 根目录的主权重哈希完全相同；该结果无效。

## 9. Final-200 协议

```bash
SERVED_MODEL_NAME=vanilla450 \
bash scripts/serve_model.sh outputs/models/grpo-vanilla-step450-deploy
```

另一个终端：

```bash
SERVED_MODEL_NAME=vanilla450 \
EVAL_OUTPUT_DIR=outputs/evaluation/final200-vanilla450 \
bash scripts/evaluate.sh final200-vanilla450
```

SFT 和 TRACE 分别使用唯一名称 `shopping-agent` 与 `trace450`，并写入独立目录。固定
协议为：

- 相同 200 个 task IDs；
- 每题一次 rollout；
- temperature 0，top-p 1；
- max steps 35，max generated tokens/turn 512；
- context 24,576，safety margin 512；
- context compaction off；
- Reward v3 为主结果，不依赖 LLM Judge。

Final-200 已被观察。上述命令只用于审计复现，不得再用于调参或 checkpoint 选择。

## 10. 最终指标与产物

机器可读指标见 `experiments/final_delivery.json`。外部最终归档应包含：

```text
outputs/evaluation/final200-sft/{trajectories.jsonl,summary.json}
outputs/evaluation/final200-vanilla450/{trajectories.jsonl,summary.json}
outputs/evaluation/final200-trace450/{trajectories.jsonl,summary.json}
outputs/models/grpo-vanilla-step450-deploy/merge_manifest.json
outputs/models/grpo-trace-step450-deploy/merge_manifest.json
```

归档 SHA-256：
`d4656fff687dc21fb0b56f0d8fea6cb834dad9b25d8792210185353aefeca125`。

已核验三份 trajectory 文件均为 200 行、200 个唯一 task ID、全部 `status=done`、全部
`reward_valid=true`。归档不包含 GRPO adapters 或完整 deploy weights。

## 11. 产物可用性边界

当前仓库中已确认存在：

- SFT adapter、merged model 与 train summary；
- 三份 TRACE proxy audit 归档；
- 三模型 Final-200 轨迹/摘要的外部归档；
- TRACE resumed-run log；
- deployment/hotfix 代码归档。

当前仓库中未发现：

- Vanilla 与 TRACE 的 selected adapter weights；
- Vanilla 完整 training diagnostics/log；
- Final deploy weight hashes。

最后使用的训练服务器当前不可连接，因此其模型、adapter 与诊断路径只能视为最后已知
位置，不能视为本次已复核。释放任何仍持久化的云存储前，应优先备份两个 selected
adapters、诊断、配置、日志和 deploy weight hashes。

## 12. 轻量验证

包装阶段允许执行：

```bash
git status --short --branch
python -m json.tool experiments/final_delivery.json >/dev/null
python -m json.tool data/evaluation/metadata.json >/dev/null
pytest -q
```

不要为了文档验证重新训练、导出/合并模型或运行 Final-200。
