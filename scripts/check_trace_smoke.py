#!/usr/bin/env python3
"""Fail closed unless a completed TRACE smoke run is safe to resume."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-step", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = args.output / "training_diagnostics.jsonl"
    if not diagnostics.is_file():
        raise SystemExit(f"missing diagnostics: {diagnostics}")
    events = []
    with diagnostics.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "optimizer_step":
                events.append(event)
    if not events or int(events[-1].get("global_step", -1)) != args.expected_step:
        raise SystemExit(
            f"smoke did not finish step {args.expected_step}: "
            f"last={events[-1].get('global_step') if events else None}"
        )
    failures = []
    eligible_trajectories = 0.0
    generated_groups = trained_groups = 0.0
    for event in events:
        step = int(event["global_step"])
        metrics = event.get("metrics") or {}
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                failures.append(f"step {step}: non-finite metric {name}={value}")
        if int(metrics.get("training/optimizer_updated", 0)) != 1:
            failures.append(f"step {step}: optimizer_updated != 1")
        for name in (
            "trajectory/infrastructure_invalid_rate",
            "trajectory/reward_unverifiable_rate",
            "trajectory/sampling_invalid_rate",
        ):
            if float(metrics.get(name, 0.0)) != 0.0:
                failures.append(f"step {step}: {name}={metrics.get(name)}")
        if float(metrics.get("trajectory/done_rate", 0.0)) < 1.0:
            failures.append(
                f"step {step}: trajectory/done_rate={metrics.get('trajectory/done_rate')}"
            )
        turn_abs_max = float(metrics.get("trace/turn_reward_abs_max", 0.0))
        if turn_abs_max > 1.000001:
            failures.append(f"step {step}: TRACE clip violated ({turn_abs_max})")
        eligible_trajectories += float(
            metrics.get("trace/eligible_trajectories", 0.0)
        )
        generated_groups += float(metrics.get("group/generated", 0.0))
        trained_groups += float(metrics.get("group/trained", 0.0))
    if eligible_trajectories <= 0:
        failures.append("TRACE never received an eligible constant-failure trajectory")
    checkpoint = args.output / f"global_step_{args.expected_step}"
    if not checkpoint.is_dir():
        failures.append(f"missing checkpoint: {checkpoint}")
    report = {
        "decision": "go" if not failures else "no_go",
        "optimizer_steps": len(events),
        "last_step": int(events[-1]["global_step"]),
        "trace_eligible_trajectories_total": eligible_trajectories,
        "generated_to_trained_group_ratio": (
            generated_groups / trained_groups if trained_groups else None
        ),
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
