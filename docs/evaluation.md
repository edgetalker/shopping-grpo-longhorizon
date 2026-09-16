# Final-200 冻结评测

## 主评测原则

最终主结果来自 ShopSimulator Environment v2.1 / Reward v3 的确定性终局，不依赖
LLM Judge。严格成功必须满足：

```text
complete gold_purchase AND reward_valid=true
```

Final-200 已完成并被观察。它不能继续用于目标设计、超参数调整、checkpoint 选择或
方法迭代。

## 数据与协议

| Item | Value |
|---|---|
| Tasks | 200 unique task IDs |
| Dataset | `data/evaluation/tasks.jsonl` |
| SHA-256 | `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5` |
| Rollouts | 1 per model per task |
| Temperature / top-p | 0 / 1 |
| Max steps | 35 |
| Max tokens per turn | 512 |
| Context | 24,576，safety margin 512 |
| Context compaction | off |
| Reward | Reward v3 |

SFT、Vanilla step 450、TRACE step 450 使用完全相同的 200 个 task IDs。三组各 200 条
轨迹均完成，无缺失、重复或运行时错误，600 条全部 `reward_valid=true`。

## 评测对象

最终评测使用：

- SFT merged model；
- SFT merged + Vanilla step-450 adapter 的独立 PEFT merge；
- SFT merged + TRACE step-450 adapter 的独立 PEFT merge；
- 三个唯一 served names：`shopping-agent`、`vanilla450`、`trace450`。

## Final-200 结果

| Model | Strict success | Mean Reward v3 | Weighted score | Mean steps | Guards |
|---|---:|---:|---:|---:|---:|
| SFT | 123/200 = 61.5% | 0.465843 | 0.750899 | 8.115 | 24 |
| Vanilla step 450 | 126/200 = 63.0% | 0.490607 | 0.747704 | 7.540 | 61 |
| TRACE step 450 | 123/200 = 61.5% | 0.464110 | 0.748079 | 7.950 | 32 |

| Model | Gold | Partial alternative | Wrong | Repeat | Max steps |
|---|---:|---:|---:|---:|---:|
| SFT | 123 | 44 | 27 | 4 | 2 |
| Vanilla | 126 | 41 | 19 | 13 | 1 |
| TRACE | 123 | 44 | 25 | 7 | 1 |

## 配对统计

相同 task IDs 构成配对二元结果，因此严格成功使用 exact McNemar 检验：

| Comparison | Delta | Gains / losses | Exact p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Vanilla − SFT | +1.5 pp | 9 / 6 | 0.607239 | [-2.5, +5.5] pp |
| TRACE − Vanilla | -1.5 pp | 3 / 6 | 0.507812 | [-4.5, +1.5] pp |
| TRACE − SFT | 0 pp | 4 / 4 | 1.0 | — |

TRACE − Vanilla mean Reward delta 为 `-0.02650`，95% interval
`[-0.07216, +0.01494]`。没有统计支持的严格成功率提升。

## 轨迹级行为

TRACE vs Vanilla 的 strict-success discordant tasks：

- gains：`2105`, `5049`, `13989`；
- losses：`7670`, `12195`, `12353`, `15496`, `15638`, `18495`。

| Behavior | Vanilla | TRACE |
|---|---:|---:|
| Purchases | 186 | 192 |
| Gold among purchases | 67.7% | 64.1% |
| Wrong-ASIN purchases | 46 | 52 |
| Budget failures | 17 | 23 |
| `back_to_search` | 21 | 3 |
| Repeat loops | 13 | 7 |
| Wrong purchases | 19 | 25 |

循环和 Guard 减少伴随更多过早错误购买，说明行为效率指标不能替代任务质量。

## 产物

权威归档包含：

```text
outputs/evaluation/final200-sft/{trajectories.jsonl,summary.json}
outputs/evaluation/final200-vanilla450/{trajectories.jsonl,summary.json}
outputs/evaluation/final200-trace450/{trajectories.jsonl,summary.json}
outputs/models/grpo-vanilla-step450-deploy/merge_manifest.json
outputs/models/grpo-trace-step450-deploy/merge_manifest.json
```

归档 SHA-256 为
`d4656fff687dc21fb0b56f0d8fea6cb834dad9b25d8792210185353aefeca125`。
它不包含 GRPO adapters 或完整 deploy weights。

机器可读最终指标见 `experiments/final_delivery.json`，复现与产物边界见
[reproducibility.md](reproducibility.md)。

## 历史材料

旧 dashboard、Judge 设计草稿以及 `experiments/{baseline,sft,grpo}` 的 step-100
summary 仅作为历史资料。它们不是本页三模型最终结论的证据源。
