# LoRA SFT

## 目的

基础模型能够生成自然语言，但不能稳定遵守 ShopSimulator 的工具协议。SFT 负责建立
后续 GRPO 所需的基础策略：合法工具调用、根据 observation 行动、选择完整 variant、
检查价格并达到有效终局。

## 最终训练输入

| Item | Value |
|---|---|
| Base model | Qwen3.5-2B |
| Dataset | `data/sft_pure_v4/all.jsonl` |
| Curriculum selector | `data/sft_curriculum/manifest.json`, stage `c` |
| Actual train / validation examples | 1,069 / 118 |
| Final-200 task-ID overlap | 0 |
| Loss mask | assistant action tokens only |

Curriculum manifest 中的原始 stage-c row 计数为 1,073/119；经过训练数据加载与有效性
过滤后，`train_summary.json` 记录的实际样本数为 1,069/118。最终报告使用后者。

旧的 `data/sft/` 800/200 split 和 staged curriculum launcher 仅用于历史实验，不是最终
SFT checkpoint 的训练来源。

## 最终配置

| Setting | Value |
|---|---|
| Max sequence length | 24,576 |
| Epochs | 1 |
| Train / eval batch | 1 / 1 |
| Gradient accumulation | 8 |
| Learning rate | `5e-5` |
| Warmup ratio | 0.03 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Precision | bf16 |
| Attention | SDPA |
| Gradient checkpointing | enabled |
| Seed | 42 |

精确命令见 [reproducibility.md](reproducibility.md)。当前包装阶段不要重新执行训练或
模型合并。

## 结果

| Metric | Value |
|---|---:|
| Train loss | 0.3983341980 |
| Runtime | 5,428.0 s / 92.9 min |
| Peak GPU memory | 68.97 GiB |
| Final-200 strict success | 123/200 = 61.5% |
| Final-200 mean Reward v3 | 0.465843 |

权威训练摘要：
`outputs/models/sft-single-pass/adapter/train_summary.json`。

关键权重哈希：

- merged model：`fe0a6e374cdbe1b401afc826254f830e96fabbff33cf84087d847760bb267c85`；
- adapter：`8885516ff96c9d91bd57f775e381157508d1fa4465a6c563e248031bac905aeb`。

## 输出契约

GRPO 从合并后的 standalone model 开始，不直接把 SFT adapter 当 base：

```text
outputs/models/sft-single-pass/merged
```

SFT 是主要能力增益来源。Vanilla 与 TRACE 的最终比较必须使用新的 Final-200 结果，
不能引用历史 `experiments/sft/summary.json` 的 60.5%。
