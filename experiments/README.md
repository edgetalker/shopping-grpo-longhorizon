# Experiments

Authoritative delivery artifacts:

```text
final_delivery.json   machine-readable SFT/Vanilla/TRACE metrics and hashes
comparison.md         human-readable final comparison and claim boundaries
```

The `baseline/`, `sft/`, and `grpo/` subdirectories contain the older
Baseline/SFT/GRPO-step100 snapshot. They are retained as historical artifacts
and must not be used as the final three-model comparison.

Large checkpoints and complete trajectories are intentionally outside Git. See
[reproducibility](../docs/reproducibility.md) for the verified archive hash,
artifact inventory, and missing-backup warning.
