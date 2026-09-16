#!/usr/bin/env python3
"""Audit TRACE targets on replayed GRPO-train trajectories without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.projection import project_observation
from shopping_grpo.environment.tools import tool_call_to_action
from shopping_grpo.training.grpo.trace_audit import (
    choose_variant,
    event_label,
    load_diagnostic_rollouts,
    paired_gold_wrong_sample,
    stratified_sample,
    summarize_paired_gold_wrong,
    summarize_variant,
    target_fingerprint,
    target_variants,
)


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_train_boundary(train_data: Path) -> str:
    metadata = json.loads((ROOT / "data/grpo/metadata.json").read_text(encoding="utf-8"))
    expected = str(metadata["train"]["parquet_sha256"])
    actual = file_sha256(train_data)
    if actual != expected:
        raise SystemExit(
            "Phase 0 accepts only the frozen GRPO training parquet: "
            f"expected SHA-256 {expected}, got {actual}"
        )
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/grpo/train.parquet")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-config", type=Path, default=ROOT / "configs/tools.json")
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--per-outcome", type=int, default=4)
    parser.add_argument(
        "--paired-gold-wrong-tasks",
        type=int,
        default=0,
        help="Phase 0b: select this many same-task gold/wrong pairs",
    )
    parser.add_argument(
        "--target-variant",
        choices=("all", "prose_json", "tool_sequence_json"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-sequence-length", type=int, default=24576)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def load_prompts(path: Path) -> dict[int, list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["prompt", "extra_info"]).to_pylist()
    prompts: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        extra = row.get("extra_info") or {}
        task_id = extra.get("task_id")
        if task_id is None:
            task_id = (extra.get("interaction_kwargs") or {}).get("task_id")
        if task_id is None:
            raise ValueError("GRPO parquet row is missing extra_info.task_id")
        prompts[int(task_id)] = [dict(message) for message in row["prompt"]]
    return prompts


def _purchase_ready(observation: str) -> bool:
    return "purchase_ready: True" in observation or "purchase_ready: true" in observation


def replay(record: dict[str, Any], prompt: list[dict[str, Any]], tokenizer, env_url: str):
    messages = [dict(message) for message in prompt]
    prefixes = [[dict(message) for message in messages]]
    turns = []
    terminal = None
    previous_action = None
    with ShopAgentEnv(base_url=env_url, timeout=60) as env:
        env.include_trace_target = True
        initial = env.reset(int(record["task_id"]))
        private_target = initial.get("_trace_target")
        if not isinstance(private_target, dict):
            raise ValueError("environment reset did not return the private TRACE target")
        if initial.get("observation_state") is not None:
            latest_observation = render_structured_observation(initial["observation_state"])
        else:
            latest_observation = str(initial.get("instruction", initial.get("observation", "")))

        for index, action in enumerate(record["actions"]):
            tool = str(action["tool"])
            parameters = dict(action.get("parameters") or {})
            call_id = f"audit-{record['trajectory_id']}-{index}"
            call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool,
                    # Hugging Face's Qwen3.5 chat template expects a mapping;
                    # OpenAI wire-format JSON strings are only used by HTTP clients.
                    "arguments": parameters,
                },
            }
            messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            purchase_ready_before = _purchase_ready(latest_observation)
            normalized_action = (tool, json.dumps(parameters, ensure_ascii=False, sort_keys=True))
            repeated = normalized_action == previous_action
            previous_action = normalized_action
            if tool == "think":
                visible = "Reasoning recorded. Continue with one environment tool call."
                result = {"done": False}
            else:
                result = env.step(tool_call_to_action(tool, parameters))
                if result.get("done"):
                    visible = "Environment terminated."
                else:
                    if result.get("observation_state") is not None:
                        raw = render_structured_observation(result["observation_state"])
                    else:
                        raw = str(result.get("instruction", result.get("observation", "")))
                    visible, _ = project_observation(
                        tool_name=tool,
                        observation=raw,
                        parameters=parameters,
                        count_tokens=lambda text: len(
                            tokenizer.encode(text, add_special_tokens=False)
                        ),
                        token_budget=1536,
                        detail_token_budget=4096,
                        generic_token_budget=768,
                        search_top_k=20,
                    )
                    latest_observation = visible
            purchase_ready_after = _purchase_ready(latest_observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool,
                    "content": visible,
                }
            )
            turns.append(
                {
                    "index": index,
                    "tool": tool,
                    "event": event_label(
                        tool,
                        parameters,
                        private_target,
                        repeated=repeated,
                        purchase_ready_before=purchase_ready_before,
                        purchase_ready_after=purchase_ready_after,
                    ),
                }
            )
            prefixes.append([dict(message) for message in messages])
            if result.get("done"):
                terminal = result
                break
    detail = (terminal or {}).get("reward_detail") or {}
    replay_valid = bool(
        terminal
        and detail.get("reward_type") == record.get("reward_type")
        and detail.get("reward_valid") is True
    )
    return {
        "private_target": private_target,
        "prefixes": prefixes,
        "turns": turns,
        "replay_valid": replay_valid,
        "replay_reward_type": detail.get("reward_type"),
    }


def load_tools(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tools = [item.get("tool_schema") for item in payload.get("tools", [])]
    if not tools or any(not isinstance(tool, dict) for tool in tools):
        raise ValueError(f"invalid tool schema config: {path}")
    return tools


def encode_jobs(
    replays,
    tokenizer,
    chat_template,
    tools,
    max_sequence_length: int,
    target_variant: str = "all",
):
    jobs = []
    fingerprints = {}
    for replay_index, replayed in enumerate(replays):
        variants = target_variants(replayed["private_target"])
        if target_variant != "all":
            variants = {target_variant: variants[target_variant]}
        for variant, target in variants.items():
            fingerprints.setdefault(variant, set()).add(target_fingerprint(target))
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            for state_index, messages in enumerate(replayed["prefixes"]):
                prefix_text = chat_template.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                budget = max_sequence_length - len(target_ids)
                if budget <= 0:
                    raise ValueError("TRACE target exceeds max sequence length")
                truncated = len(prefix_ids) > budget
                prefix_ids = list(prefix_ids[-budget:])
                jobs.append(
                    {
                        "replay_index": replay_index,
                        "trajectory_id": replayed["trajectory_id"],
                        "variant": variant,
                        "state_index": state_index,
                        "target_sha256": target_fingerprint(target),
                        "input_ids": [*prefix_ids, *target_ids],
                        "target_start": len(prefix_ids),
                        "target_length": len(target_ids),
                        "prefix_truncated": truncated,
                    }
                )
    return jobs, fingerprints


def load_score_cache(path: Path, jobs) -> int:
    """Reuse every matching trajectory/target/state score in an existing cache."""
    if not path.is_file():
        return 0
    cached = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                row["trajectory_id"],
                row["variant"],
                int(row["state_index"]),
                row["target_sha256"],
            )
            cached[key] = float(row["mean_log_prob"])
    reused = 0
    for job in jobs:
        key = (
            job["trajectory_id"],
            job["variant"],
            int(job["state_index"]),
            job["target_sha256"],
        )
        if key in cached:
            job["mean_log_prob"] = cached[key]
            reused += 1
    return reused


def save_score_cache(path: Path, jobs) -> None:
    """Persist scores without actor-visible text, token IDs, or private targets."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for job in jobs:
            row = {
                "trajectory_id": job["trajectory_id"],
                "variant": job["variant"],
                "state_index": int(job["state_index"]),
                "target_sha256": job["target_sha256"],
                "mean_log_prob": float(job["mean_log_prob"]),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def score_jobs(jobs, model, *, batch_size: int):
    import torch

    jobs.sort(key=lambda job: len(job["input_ids"]))
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        maximum = max(len(job["input_ids"]) for job in batch)
        input_ids = []
        attention = []
        for job in batch:
            padding = maximum - len(job["input_ids"])
            input_ids.append([*job["input_ids"], *([model.config.pad_token_id] * padding)])
            attention.append([1] * len(job["input_ids"]) + [0] * padding)
        ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
        mask = torch.tensor(attention, dtype=torch.long, device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        for row, job in enumerate(batch):
            first = int(job["target_start"])
            length = int(job["target_length"])
            token_ids = ids[row, first : first + length]
            positions = torch.arange(first - 1, first + length - 1, device=model.device)
            # Only materialize FP32 logits at the short target positions.  Casting
            # the full [batch, sequence, vocab] tensor would exceed 96 GB for
            # Qwen3.5's large vocabulary and long shopping contexts.
            target_logits = logits[row, positions, :].float()
            values = target_logits.gather(1, token_ids[:, None]).squeeze(1)
            values = values - torch.logsumexp(target_logits, dim=-1)
            job["mean_log_prob"] = float(values.mean().cpu())
        del logits, ids, mask


def assemble_rows(selected, replays, jobs, target_variant: str = "all"):
    scores: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for job in jobs:
        scores.setdefault((job["replay_index"], job["variant"]), []).append(job)
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for replay_index, (record, replayed) in enumerate(zip(selected, replays, strict=True)):
        reward = record["reward"]
        variants = target_variants(replayed["private_target"])
        if target_variant != "all":
            variants = {target_variant: variants[target_variant]}
        for variant in variants:
            state_jobs = sorted(
                scores[(replay_index, variant)], key=lambda job: job["state_index"]
            )
            state_scores = [float(job["mean_log_prob"]) for job in state_jobs]
            turns = []
            for turn, left, right in zip(
                replayed["turns"],
                state_scores[:-1],
                state_scores[1:],
                strict=True,
            ):
                turns.append(
                    {
                        **turn,
                        "score_before": left,
                        "score_after": right,
                        "score_delta": right - left,
                    }
                )
            row = {
                "trajectory_id": record["trajectory_id"],
                "task_id": int(record["task_id"]),
                "global_step": int(record["global_step"]),
                "uid": str(record.get("uid")),
                "rollout_index": int(record.get("rollout_index", 0)),
                "reward_type": record["reward_type"],
                "terminal_utility": float(reward["terminal_utility"]),
                "target_variant": variant,
                "target_sha256": target_fingerprint(
                    target_variants(replayed["private_target"])[variant]
                ),
                "target_visible_to_actor": False,
                "replay_valid": replayed["replay_valid"],
                "replay_reward_type": replayed["replay_reward_type"],
                "initial_score": state_scores[0],
                "final_score": state_scores[-1],
                "score_delta": state_scores[-1] - state_scores[0],
                "prefix_truncated": any(job["prefix_truncated"] for job in state_jobs),
                "turns": turns,
            }
            by_variant.setdefault(variant, []).append(row)
    return by_variant


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {args.output}")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    started = time.monotonic()
    train_sha256 = validate_train_boundary(args.train_data)
    diagnostics_sha256 = file_sha256(args.diagnostics)
    records = load_diagnostic_rollouts(args.diagnostics)
    if args.paired_gold_wrong_tasks:
        selected, available = paired_gold_wrong_sample(
            records, task_count=args.paired_gold_wrong_tasks, seed=args.seed
        )
        audit_mode = "paired_gold_wrong_phase0b"
    else:
        selected, available = stratified_sample(
            records, per_outcome=args.per_outcome, seed=args.seed
        )
        audit_mode = "stratified_phase0"
    prompts = load_prompts(args.train_data)

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
    )

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    if is_multimodal:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        tokenizer = processor.tokenizer
        chat_template = processor
        model_class = AutoModelForMultimodalLM
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        chat_template = tokenizer
        model_class = AutoModelForCausalLM
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tools = load_tools(args.tool_config)
    replays = []
    for index, record in enumerate(selected, start=1):
        print(f"replay {index}/{len(selected)}: {record['trajectory_id']}", flush=True)
        replayed = replay(record, prompts[int(record["task_id"])], tokenizer, args.env_url)
        replayed["trajectory_id"] = record["trajectory_id"]
        replays.append(replayed)

    jobs, fingerprints = encode_jobs(
        replays,
        tokenizer,
        chat_template,
        tools,
        args.max_sequence_length,
        args.target_variant,
    )
    score_cache = Path(str(args.output) + ".score-cache.jsonl")
    scoring_started = time.monotonic()
    reused_scores = load_score_cache(score_cache, jobs)
    pending_jobs = [job for job in jobs if "mean_log_prob" not in job]
    if not pending_jobs:
        print(f"reusing complete TRACE score cache: {score_cache}", flush=True)
        scoring_seconds = 0.0
        peak_memory = 0
    else:
        model = model_class.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model.eval()
        model.config.pad_token_id = tokenizer.pad_token_id
        torch.cuda.reset_peak_memory_stats()
        if reused_scores:
            print(
                f"reusing {reused_scores} cached scores; scoring {len(pending_jobs)} new jobs",
                flush=True,
            )
        score_jobs(pending_jobs, model, batch_size=args.batch_size)
        scoring_seconds = time.monotonic() - scoring_started
        peak_memory = int(torch.cuda.max_memory_allocated())
        save_score_cache(score_cache, jobs)
        print(f"saved TRACE score cache: {score_cache}", flush=True)
    by_variant = assemble_rows(selected, replays, jobs, args.target_variant)
    if args.paired_gold_wrong_tasks:
        summaries = {
            name: summarize_paired_gold_wrong(
                rows, minimum_pairs=args.paired_gold_wrong_tasks
            )
            for name, rows in by_variant.items()
        }
    else:
        summaries = {name: summarize_variant(rows) for name, rows in by_variant.items()}
    chosen, passed = choose_variant(summaries)

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "trace_proxy_scores.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for variant in sorted(by_variant):
            for row in by_variant[variant]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "audit_mode": audit_mode,
        "decision": "go" if passed else "no_go",
        "selected_target_variant": chosen,
        "data_boundary": {
            "source": "GRPO training diagnostics only",
            "evaluation_data_used": False,
            "private_target_visible_to_actor": False,
            "raw_private_targets_serialized": False,
        },
        "inputs": {
            "diagnostics": str(args.diagnostics.resolve()),
            "diagnostics_sha256": diagnostics_sha256,
            "train_data": str(args.train_data.resolve()),
            "train_data_sha256": train_sha256,
            "model": str(args.model.resolve()),
            "seed": args.seed,
            "per_outcome": args.per_outcome,
            "paired_gold_wrong_tasks": args.paired_gold_wrong_tasks,
            "target_variant": args.target_variant,
        },
        "available_outcome_counts": available,
        "selected_trajectory_count": len(selected),
        "target_fingerprint_counts": {
            name: len(values) for name, values in fingerprints.items()
        },
        "runtime": {
            "score_job_count": len(jobs),
            "score_cache_reused_count": reused_scores,
            "score_job_new_count": len(pending_jobs),
            "scoring_seconds": scoring_seconds,
            "total_seconds": time.monotonic() - started,
            "peak_cuda_bytes": peak_memory,
            "batch_size": args.batch_size,
            "score_cache": str(score_cache.resolve()),
        },
        "variants": summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
