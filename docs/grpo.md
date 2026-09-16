# GRPO 与 failure-only 回合信用

## 目标

SFT 先建立工具协议与合法终局能力。GRPO 再在 ShopSimulator 中生成新轨迹，并使用
Reward v3 优化约束满足和策略选择。Reward v3 是唯一权威终局 reward，不使用额外
LLM-as-a-Judge reward model。

## Vanilla 配置

| Setting | Value |
|---|---|
| Initialization | `outputs/models/sft-single-pass/merged` |
| veRL / estimator | 0.8.0 / GRPO |
| Rollouts per prompt | 4 |
| Temperature / top-p | 0.7 / 0.9 |
| Prompt / PPO mini / micro batch | 2 / 2 / 1 |
| Learning rate | `1e-6`，3% warmup |
| LoRA rank / alpha | 16 / 32 |
| Maximum model length | 24,576 |
| Optimizer steps | 500 |
| Save / validation frequency | 50 / 50 |
| Reward KL / actor KL / entropy / length shaping | off / off / off / off |
| Dynamic generation batches | at most 3 per attempt |

`configs/grpo.yaml` 是当前配置源。运行前只解析命令：

```bash
bash scripts/grpo.sh \
  --model outputs/models/sft-single-pass/merged \
  --output outputs/models/grpo-vanilla-500-entropy-off-v1 \
  --experiment-name grpo-vanilla-500-entropy-off-v1 \
  --dry-run
```

## 诊断

Vanilla 的 500 次更新生成 2,270 个组 / 9,080 条轨迹，但只训练 1,000 个组 /
4,000 条轨迹。47.8% 的生成组 reward 恒定，导致组相对 advantage 为零。动态采样能
避免零梯度 update，却需要额外生成，且让有效训练分布偏向更容易产生 reward 差异的
任务。

## TRACE-inspired failure-only 分支

选择逻辑：

1. terminal-varying valid groups：完全按 Vanilla 处理；
2. valid exact-tie groups with no purchase success：保留并计算回合信用；
3. all-success constant groups：丢弃；
4. sampling-invalid / unverifiable groups：丢弃。

训练端私有 target：

```text
最终应购买商品：{"asin":"...","options":{...}}
```

它不进入 actor observation 或公共诊断。参数：

| Parameter | Value |
|---|---:|
| gate | `exact_tie_failure_only` |
| epsilon | 0.1 |
| horizon | 3 |
| discount | 0.8 |
| terminal / outcome / turn weight | 2.0 / 1.0 / 0.2 |
| centered credit clip | `[-1, 1]` |
| scorer max sequence length | 24,576 |

运行时必须同时打开配置与环境保护开关：

```bash
SHOPPING_TRACE_FAILURE_ONLY=true \
bash scripts/grpo.sh \
  --model outputs/models/sft-single-pass/merged \
  --output outputs/models/grpo_trace_failure_only_v1 \
  --experiment-name grpo_trace_failure_only_v1 \
  --dry-run \
  -- shopping_trace.enable=true
```

当前处于文档包装阶段，不应移除 `--dry-run` 启动新训练。

## 与论文 TRACE 的边界

本实现不是 faithful/global TRACE：它只覆盖 exact-tie failure groups，额外做 UID 内
中心化和裁剪，使用 group size 4、小 batch、LoRA rank 16，并把结构化购买字符串作为
scorer target。正式名称应包含 `TRACE-inspired` 和 `failure-only`。

## 结果

| Metric | Vanilla | TRACE |
|---|---:|---:|
| Generated groups | 2,270 | 1,806 |
| Generated trajectories | 9,080 | 7,224 |
| Generated/trained ratio | 2.270 | 1.806 |
| Utilization | 44.05% | 55.37% |
| Skipped attempts | 56 | 17 |
| Selected validation reward | 0.327932 | 0.355292 |
| Final-200 strict success | 63.0% | 61.5% |

利用率提升成立，但最终任务质量提升不成立。完整解释见
[项目罗盘](project-compass.md)，精确命令和导出流程见
[复现实验说明](reproducibility.md)。

## 实现位置

```text
src/shopping_grpo/training/grpo/trace.py
src/shopping_grpo/training/grpo/compat.py
src/shopping_grpo/training/grpo/dynamic_sampling.py
src/shopping_grpo/training/grpo/adapter/
scripts/check_grpo_runtime.py
scripts/check_trace_smoke.py
scripts/audit_trace_proxy.py
```

相关测试覆盖回合信用、动态采样、veRL 适配、proxy audit 和公共入口。
