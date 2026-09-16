# SFT → Vanilla GRPO → Failure-only TRACE

这是最终权威比较。`baseline/`、`sft/`、`grpo/` 子目录中的旧 summary 是早期
Baseline/SFT/GRPO-step100 快照，仅保留用于历史审计。

## Protocol

- ShopSimulator Environment v2.1 / Reward v3；
- 相同 200 个冻结 task IDs；
- 每个模型每题一次确定性 rollout；
- `temperature=0`，`top_p=1`，`max_steps=35`；
- context 24,576，context compaction off；
- 600/600 trajectories 为 `status=done` 且 `reward_valid=true`；
- step 450 使用 50 题 validation 选择，Final-200 不参与 checkpoint 选择。

## Final-200

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

## Paired statistics

| Comparison | Δ strict success | Gains / losses | Exact McNemar p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Vanilla − SFT | +1.5 pp | 9 / 6 | 0.607239 | [-2.5, +5.5] pp |
| TRACE − Vanilla | -1.5 pp | 3 / 6 | 0.507812 | [-4.5, +1.5] pp |
| TRACE − SFT | 0 pp | 4 / 4 | 1.0 | — |

TRACE − Vanilla additional paired results:

- mean Reward delta `-0.02650`, 95% interval `[-0.07216, +0.01494]`；
- weighted-score delta `+0.00038`；
- mean-step delta `+0.41`；
- Guard-rejection delta `-0.145` per task。

Neither Vanilla nor failure-only TRACE has a statistically supported
strict-success gain.

## Generation utilization

Both GRPO arms completed 500 optimizer updates and trained 1,000 groups / 4,000
trajectories.

| Metric | Vanilla | TRACE | Change |
|---|---:|---:|---:|
| Generated groups | 2,270 | 1,806 | -20.4% |
| Generated trajectories | 9,080 | 7,224 | -20.4% |
| Generated/trained ratio | 2.270 | 1.806 | -20.4% |
| Utilization | 44.05% | 55.37% | +11.32 pp |
| Skipped attempts | 56 | 17 | -69.6% |

This establishes improved generation utilization, not measured wall-clock or
GPU-hour savings.

## Trajectory diagnosis

TRACE vs Vanilla discordant tasks:

- gains: `2105`, `5049`, `13989`；
- losses: `7670`, `12195`, `12353`, `15496`, `15638`, `18495`。

TRACE reduced repeat loops `13 → 7`, Guard rejections `61 → 32`, and
`back_to_search` actions `21 → 3`, but increased wrong purchases `19 → 25` and
budget failures `17 → 23`. Fewer loops partly represent earlier wrong
commitment, not a pure capability gain.

## Interpretation

SFT supplies the dominant tool-use and terminal-behavior capability. Vanilla
GRPO gives a small, non-significant held-out gain. Failure-only TRACE rescues
otherwise discarded groups and improves rollout utilization, but its target-
string potential does not encode full purchase readiness.

The correct final claim is:

> TRACE improved generation utilization and changed behavior, but did not
> produce a statistically supported strict-success improvement over Vanilla or
> SFT.

See [project compass](../docs/project-compass.md) for the mechanism and
[reproducibility](../docs/reproducibility.md) for hashes and artifact boundaries.
