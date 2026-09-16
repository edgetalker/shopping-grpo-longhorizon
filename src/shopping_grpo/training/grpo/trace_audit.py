"""Pure helpers for the train-only TRACE proxy audit."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from shopping_grpo.training.grpo.trace import canonical_purchase_target


AUDIT_OUTCOMES = (
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "wrong_purchase",
    "repeat_loop",
    "max_steps",
)


def load_diagnostic_rollouts(path: str | Path) -> list[dict[str, Any]]:
    """Load public rollout records without accepting evaluation artifacts."""
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid diagnostic JSON at line {line_number}") from exc
            if event.get("event") != "generation_batch":
                continue
            global_step = int(event.get("global_step", -1))
            for rollout in event.get("rollouts") or []:
                if not isinstance(rollout, Mapping):
                    raise ValueError(
                        f"generation_batch rollout at line {line_number} is not an object"
                    )
                item = dict(rollout)
                item["global_step"] = global_step
                item["trajectory_id"] = (
                    f"step-{global_step}:uid-{item.get('uid')}:rollout-"
                    f"{item.get('rollout_index')}"
                )
                records.append(item)
    if not records:
        raise ValueError(f"no generation_batch rollouts found in {source}")
    return records


def audit_eligible(record: Mapping[str, Any]) -> bool:
    """Select replayable, valid trajectories from the completed vanilla run."""
    reward = record.get("reward")
    return bool(
        isinstance(reward, Mapping)
        and record.get("reward_valid") is True
        and not record.get("infrastructure_invalid")
        and not reward.get("sampling_invalid")
        and int(record.get("guard_rejections", 0)) == 0
        and int(record.get("context_compactions", 0)) == 0
        and isinstance(record.get("actions"), list)
        and record.get("reward_type") in AUDIT_OUTCOMES
    )


def stratified_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    per_outcome: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample each outcome, prioritizing tasks that support paired comparisons."""
    if per_outcome <= 0:
        raise ValueError("per_outcome must be positive")
    eligible = [dict(record) for record in records if audit_eligible(record)]
    outcomes_by_task: dict[int, set[str]] = defaultdict(set)
    for record in eligible:
        outcomes_by_task[int(record["task_id"])].add(str(record["reward_type"]))

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    available: dict[str, int] = {}
    for outcome in AUDIT_OUTCOMES:
        bucket = [record for record in eligible if record["reward_type"] == outcome]
        rng.shuffle(bucket)

        def priority(record: Mapping[str, Any]) -> tuple[int, str]:
            task_outcomes = outcomes_by_task[int(record["task_id"])]
            if outcome == "gold_purchase":
                paired = "wrong_purchase" in task_outcomes
            elif outcome == "wrong_purchase":
                paired = bool(
                    {"gold_purchase", "valid_alternative_purchase"} & task_outcomes
                )
            elif outcome == "valid_alternative_purchase":
                paired = "wrong_purchase" in task_outcomes
            else:
                paired = False
            return (0 if paired else 1, str(record["trajectory_id"]))

        bucket.sort(key=priority)
        available[outcome] = len(bucket)
        selected.extend(bucket[:per_outcome])
    if not selected:
        raise ValueError("diagnostics contain no audit-eligible trajectories")
    return selected, available


def paired_gold_wrong_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    task_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select one gold/wrong trajectory from each comparable train task."""
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    eligible = [dict(record) for record in records if audit_eligible(record)]
    by_task: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in eligible:
        by_task[int(record["task_id"])][str(record["reward_type"])].append(record)
    candidates = [
        task_id
        for task_id, outcomes in by_task.items()
        if outcomes.get("gold_purchase") and outcomes.get("wrong_purchase")
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen_tasks = candidates[:task_count]
    selected = []
    for task_id in chosen_tasks:
        outcomes = by_task[task_id]
        for outcome in ("gold_purchase", "wrong_purchase"):
            bucket = sorted(outcomes[outcome], key=lambda row: str(row["trajectory_id"]))
            selected.append(dict(rng.choice(bucket)))
    if len(chosen_tasks) < task_count:
        raise ValueError(
            f"requested {task_count} paired tasks but only {len(candidates)} are available"
        )
    return selected, {
        "paired_task_count": len(candidates),
        "selected_paired_task_count": len(chosen_tasks),
    }


def target_variants(private_target: Mapping[str, Any]) -> dict[str, str]:
    """Return the legacy prose target and a real-tool-schema-aligned target."""
    asin = private_target.get("asin")
    options = private_target.get("options")
    prose = canonical_purchase_target(asin, options)
    calls: list[dict[str, Any]] = [
        {"name": "open_product", "arguments": {"asin": str(asin).strip()}}
    ]
    if isinstance(options, Mapping):
        option_values = [options[key] for key in sorted(options, key=str)]
    elif isinstance(options, (list, tuple)):
        option_values = list(options)
    elif options is None:
        option_values = []
    else:
        raise ValueError("TRACE target options must be an object or array")
    calls.extend(
        {"name": "select_option", "arguments": {"value": str(value)}}
        for value in option_values
    )
    calls.append({"name": "buy_now", "arguments": {}})
    tool_sequence = json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"prose_json": prose, "tool_sequence_json": tool_sequence}


def target_fingerprint(target: str) -> str:
    return hashlib.sha256(target.encode("utf-8")).hexdigest()


def event_label(
    tool: str,
    parameters: Mapping[str, Any],
    private_target: Mapping[str, Any],
    *,
    repeated: bool,
    purchase_ready_before: bool,
    purchase_ready_after: bool,
) -> str:
    """Assign train-only event labels without serializing the private target."""
    if repeated:
        return "repeated_action"
    if not purchase_ready_before and purchase_ready_after:
        return "purchase_ready_transition"
    if tool == "open_product":
        return (
            "open_gold_product"
            if str(parameters.get("asin", "")) == str(private_target.get("asin", ""))
            else "open_non_gold_product"
        )
    if tool == "select_option":
        options = private_target.get("options")
        values = options.values() if isinstance(options, Mapping) else options or []
        return (
            "select_target_option"
            if str(parameters.get("value", "")) in {str(value) for value in values}
            else "select_other_option"
        )
    return str(tool)


def _pairwise_agreement(
    rows: Sequence[Mapping[str, Any]], positive: str, negative: str
) -> dict[str, float | int | None]:
    by_task: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task[int(row["task_id"])][str(row["reward_type"])].append(
            float(row["score_delta"])
        )
    wins = ties = pairs = 0
    for outcomes in by_task.values():
        for left in outcomes.get(positive, []):
            for right in outcomes.get(negative, []):
                pairs += 1
                wins += left > right
                ties += left == right
    agreement = (wins + 0.5 * ties) / pairs if pairs else None
    return {"pairs": pairs, "wins": wins, "ties": ties, "agreement": agreement}


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    left = math.sqrt(sum((x - mx) ** 2 for x in xs))
    right = math.sqrt(sum((y - my) ** 2 for y in ys))
    return numerator / (left * right) if left and right else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_variant(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute predeclared train-side directionality and numerical gates."""
    valid = [row for row in rows if row.get("replay_valid") is True]
    deltas = [float(row["score_delta"]) for row in valid]
    utilities = [float(row["terminal_utility"]) for row in valid]
    turn_deltas = [
        float(turn["score_delta"])
        for row in valid
        for turn in row.get("turns", [])
    ]
    event_values: dict[str, list[float]] = defaultdict(list)
    for row in valid:
        for turn in row.get("turns", []):
            event_values[str(turn["event"])].append(float(turn["score_delta"]))

    gold_wrong = _pairwise_agreement(valid, "gold_purchase", "wrong_purchase")
    alternative_wrong = _pairwise_agreement(
        valid, "valid_alternative_purchase", "wrong_purchase"
    )
    correlation = _pearson(utilities, deltas)
    positive_events = [
        value
        for label in (
            "open_gold_product",
            "select_target_option",
            "purchase_ready_transition",
        )
        for value in event_values.get(label, [])
    ]
    no_progress_events = [
        *event_values.get("repeated_action", []),
        *event_values.get("open_non_gold_product", []),
        *event_values.get("select_other_option", []),
    ]
    finite = all(math.isfinite(value) for value in [*deltas, *turn_deltas])
    gold_ok = bool(
        gold_wrong["pairs"]
        and float(gold_wrong["agreement"]) >= 0.60
    )
    alternative_ok = bool(
        alternative_wrong["pairs"]
        and float(alternative_wrong["agreement"]) > 0.50
    )
    event_ok = bool(
        positive_events
        and (
            not no_progress_events
            or mean(positive_events) > mean(no_progress_events)
        )
    )
    p99_abs = _quantile([abs(x) for x in turn_deltas], 0.99)
    numeric_ok = bool(finite and turn_deltas and (p99_abs or 0) < 10)
    passed = bool(
        valid
        and gold_ok
        and alternative_ok
        and correlation is not None
        and correlation > 0
        and event_ok
        and numeric_ok
    )
    return {
        "trajectory_count": len(valid),
        "replay_invalid_count": len(rows) - len(valid),
        "gold_vs_wrong": gold_wrong,
        "alternative_vs_wrong": alternative_wrong,
        "terminal_utility_correlation": correlation,
        "event_mean_delta": {
            label: mean(values) for label, values in sorted(event_values.items()) if values
        },
        "positive_event_mean": mean(positive_events) if positive_events else None,
        "no_progress_event_mean": mean(no_progress_events) if no_progress_events else None,
        "numerical": {
            "finite": finite,
            "turn_count": len(turn_deltas),
            "mean": mean(turn_deltas) if turn_deltas else None,
            "p95_abs": _quantile([abs(value) for value in turn_deltas], 0.95),
            "p99_abs": p99_abs,
            "min": min(turn_deltas) if turn_deltas else None,
            "max": max(turn_deltas) if turn_deltas else None,
        },
        "gates": {
            "gold_vs_wrong_at_least_60pct": gold_ok,
            "alternative_vs_wrong_above_chance": alternative_ok,
            "positive_terminal_association": correlation is not None and correlation > 0,
            "event_alignment": event_ok,
            "numerically_controlled": numeric_ok,
        },
        "passed": passed,
    }


def one_sided_binomial_pvalue(wins: int, trials: int) -> float | None:
    """Exact P[X >= wins] for X ~ Binomial(trials, 0.5)."""
    if trials <= 0:
        return None
    if wins < 0 or wins > trials:
        raise ValueError("binomial wins must be in [0, trials]")
    return sum(math.comb(trials, value) for value in range(wins, trials + 1)) / (
        2**trials
    )


def summarize_paired_gold_wrong(
    rows: Sequence[Mapping[str, Any]], *, minimum_pairs: int = 20
) -> dict[str, Any]:
    """Apply the amended Phase 0b gates to paired train-only trajectories."""
    summary = summarize_variant(rows)
    comparison = summary["gold_vs_wrong"]
    pairs = int(comparison["pairs"])
    decisive_wins = int(comparison["wins"])
    pvalue = one_sided_binomial_pvalue(decisive_wins, pairs)
    paired_ok = bool(
        pairs >= minimum_pairs
        and comparison["agreement"] is not None
        and float(comparison["agreement"]) >= 0.60
        and pvalue is not None
        and pvalue < 0.05
    )
    summary["phase0b"] = {
        "minimum_pairs": int(minimum_pairs),
        "one_sided_exact_binomial_pvalue": pvalue,
    }
    summary["gates"]["paired_gold_wrong_significant"] = paired_ok
    summary["gates"]["alternative_vs_wrong_above_chance"] = None
    summary["passed"] = bool(
        paired_ok
        and summary["gates"]["positive_terminal_association"]
        and summary["gates"]["event_alignment"]
        and summary["gates"]["numerically_controlled"]
    )
    return summary


def choose_variant(summaries: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, bool]:
    passing = [name for name, summary in summaries.items() if summary.get("passed")]
    if not passing:
        return None, False

    def score(name: str) -> float:
        summary = summaries[name]
        gold = summary["gold_vs_wrong"].get("agreement") or 0.0
        alternative = summary["alternative_vs_wrong"].get("agreement") or 0.0
        correlation = summary.get("terminal_utility_correlation") or 0.0
        return float(gold) + float(alternative) + float(correlation)

    return max(passing, key=score), True
