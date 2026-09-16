# Shopping GRPO

An end-to-end post-training, credit-assignment, and auditable evaluation project
for long-horizon shopping agents.

The repository supports one workflow:

```text
Baseline → SFT → GRPO → Evaluation
```

The runtime contract is fixed to ShopSimulator Environment v2.1, Reward v3,
observation v2, and tool schema v2. This is not a claim of a new general RL
algorithm. It studies a narrower question: can conservative turn-level credit
recover exact-tie failure groups that vanilla GRPO would otherwise discard?

[中文](README.md) · [Project compass](docs/project-compass.md) ·
[Reproduction](docs/reproducibility.md)

## Fork and individual contributions

This repository is forked from
[`YYHDBL/shopping-grpo-longhorizon`](https://github.com/YYHDBL/shopping-grpo-longhorizon).
The experiment work starts from commit
`dc6c9af8e48c8e6325101b39d1d20060f69e4221`.

The main contributions in this fork are the Environment v2.1 / Reward v3
runtime repairs, the SFT → online GRPO pipeline, bounded dynamic sampling and
diagnostics, TRACE-inspired failure-only turn credit, and paired Final-200
analysis. The upstream provenance and the work added in this fork remain
auditable through the modular commit history.

## Result in one paragraph

LoRA SFT supplied most of the shopping capability. Vanilla GRPO produced only a
small, statistically unsupported improvement on the frozen 200-task test set.
The implemented **TRACE-inspired exact-tie failure-only turn credit for Shopping
GRPO** reduced generated trajectories by 20.4% for the same number of trained
groups, but it did not improve held-out strict success. Full trajectories explain
the mismatch: predicting a canonical ASIN/options target is not the same as being
ready to purchase under budget, option, page-state, and action-legality constraints.

## Final-200 results

All models used the same 200 task IDs and one deterministic rollout per task:
`temperature=0`, `top_p=1`, `max_steps=35`, context 24,576, and no compaction.
All 600 trajectories completed with `reward_valid=true`.

| Model | Strict success | Mean Reward v3 | Weighted score | Mean steps | Guard rejections |
|---|---:|---:|---:|---:|---:|
| SFT | 123/200 = 61.5% | 0.465843 | 0.750899 | 8.115 | 24 |
| Vanilla GRPO step 450 | 126/200 = 63.0% | 0.490607 | 0.747704 | 7.540 | 61 |
| TRACE failure-only step 450 | 123/200 = 61.5% | 0.464110 | 0.748079 | 7.950 | 32 |

Paired comparisons:

- Vanilla vs SFT: `+1.5 pp`, 9 gains / 6 losses, exact McNemar `p=0.607`;
- TRACE vs Vanilla: `-1.5 pp`, 3 gains / 6 losses, exact McNemar `p=0.508`;
- TRACE vs SFT: `0 pp`, 4 gains / 4 losses, exact McNemar `p=1.0`.

Neither GRPO arm has a statistically supported strict-success gain.

## Rollout utilization

Both GRPO arms completed 500 optimizer updates and trained on 1,000 groups /
4,000 trajectories.

| Metric | Vanilla | TRACE failure-only | Change |
|---|---:|---:|---:|
| Generated groups | 2,270 | 1,806 | -20.4% |
| Generated trajectories | 9,080 | 7,224 | -20.4% |
| Generated/trained group ratio | 2.270 | 1.806 | -20.4% |
| Training utilization | 44.05% | 55.37% | +11.32 pp |
| Skipped update attempts | 56 | 17 | -69.6% |

This is a generation-utilization result, not a measured 20.4% wall-clock or
GPU-hour saving. Reference-model scoring adds cost.

## Method scope

The method activates only for valid, no-success groups whose terminal rewards
are exactly tied. Terminal-varying groups remain vanilla GRPO, while all-success
and sampling-invalid groups are dropped.

```text
private target: 最终应购买商品：{"asin":"...","options":{...}}
epsilon=0.1, horizon=3, discount=0.8
terminal/outcome/turn weight = 2.0/1.0/0.2
centered turn-credit clip = [-1, 1]
```

It is not a faithful or global TRACE reproduction. It differs in gating,
within-group centering and clipping, group size, batch size, LoRA training, and
the structured purchase target. The official TRACE paper is
[arXiv:2607.13988](https://arxiv.org/abs/2607.13988).

## Why task quality did not improve

TRACE reduced repeat loops from 13 to 7 and Guard rejections from 61 to 32, but
increased purchases from 186 to 192, wrong purchases from 19 to 25, and budget
failures from 17 to 23. Only one of the 13 Vanilla repeat-loop tasks became a
gold success under TRACE; many others became non-gold purchases.

The scorer does not explicitly model category validity, resolved variant price,
required option completion, evidence completeness, current-page reachability,
or whether the policy should keep exploring instead of calling `buy_now`.
A future method should use a new untouched split and a constraint-aware
purchase-readiness potential.

## Data boundaries

- GRPO train / validation: 1,000 / 50 tasks;
- actual SFT train / validation examples: 1,069 / 118;
- Final-200 has zero task-ID overlap with SFT and GRPO train/validation;
- PureV4 SFT and GRPO train have six historical task-ID overlaps (0.6%); both
  GRPO arms use the same frozen train set;
- Final-200 has now been observed and must not be used for further tuning,
  checkpoint selection, or method iteration.

## Reproduction and evidence

The exact SFT/GRPO commands, dependency versions, checkpoint-selection rule,
hashes, model export procedure, and artifact boundaries are in
[docs/reproducibility.md](docs/reproducibility.md). The machine-readable final
metrics are in [experiments/final_delivery.json](experiments/final_delivery.json).

During packaging, only lightweight checks should run:

```bash
git status --short --branch
shasum -a 256 data/grpo/train.parquet \
  data/grpo/validation.parquet data/evaluation/tasks.jsonl
pytest -q
```

Do not rerun training, model merging, or Final-200 merely to validate the docs.

## Acknowledgements

Built on [ShopSimulator](https://arxiv.org/abs/2601.18225),
[veRL](https://github.com/verl-project/verl),
[Qwen](https://github.com/QwenLM/Qwen3), and the TRACE work.
