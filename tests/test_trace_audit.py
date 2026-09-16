from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.training.grpo.trace_audit import (
    audit_eligible,
    choose_variant,
    load_diagnostic_rollouts,
    paired_gold_wrong_sample,
    stratified_sample,
    summarize_variant,
    summarize_paired_gold_wrong,
    target_variants,
)
from scripts.audit_trace_proxy import assemble_rows, encode_jobs


def rollout(outcome, task_id, rollout_index=0, utility=0.0):
    return {
        "uid": f"uid-{task_id}",
        "rollout_index": rollout_index,
        "task_id": task_id,
        "reward_type": outcome,
        "reward_valid": True,
        "infrastructure_invalid": False,
        "guard_rejections": 0,
        "context_compactions": 0,
        "actions": [{"tool": "search_products", "parameters": {"query": "x"}}],
        "reward": {
            "terminal_utility": utility,
            "sampling_invalid": False,
        },
    }


class TraceAuditTest(unittest.TestCase):
    def test_encode_jobs_uses_replayed_trajectory_identity(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return list(range(1, len(text) + 1))

        class Template:
            def apply_chat_template(self, messages, **_kwargs):
                return "|".join(message["role"] for message in messages)

        replays = [
            {
                "trajectory_id": "trajectory-1",
                "private_target": {"asin": "123", "options": {}},
                "prefixes": [[{"role": "user", "content": "query"}]],
            }
        ]
        jobs, _ = encode_jobs(replays, Tokenizer(), Template(), [], 4096)
        self.assertTrue(jobs)
        self.assertEqual({job["trajectory_id"] for job in jobs}, {"trajectory-1"})

    def test_loader_and_stratified_sample_use_generation_batches_only(self):
        rows = [
            {"event": "optimizer_step", "global_step": 1},
            {
                "event": "generation_batch",
                "global_step": 3,
                "rollouts": [
                    rollout("gold_purchase", 7, utility=1.0),
                    rollout("wrong_purchase", 7, 1, -0.85),
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            loaded = load_diagnostic_rollouts(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["global_step"], 3)
        selected, available = stratified_sample(loaded, per_outcome=1, seed=1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(available["gold_purchase"], 1)

    def test_invalid_or_guard_rejected_rollout_is_not_eligible(self):
        record = rollout("wrong_purchase", 1, utility=-0.85)
        self.assertTrue(audit_eligible(record))
        record["guard_rejections"] = 1
        self.assertFalse(audit_eligible(record))
        record["guard_rejections"] = 0
        record["reward"]["sampling_invalid"] = True
        self.assertFalse(audit_eligible(record))

    def test_paired_sampler_selects_same_task_gold_and_wrong(self):
        records = []
        for task_id in range(25):
            gold = rollout("gold_purchase", task_id, utility=1.0)
            wrong = rollout("wrong_purchase", task_id, 1, -0.85)
            for item in (gold, wrong):
                item["global_step"] = task_id
                item["trajectory_id"] = (
                    f"step-{task_id}:{item['reward_type']}:{item['rollout_index']}"
                )
                records.append(item)
        selected, available = paired_gold_wrong_sample(
            records, task_count=20, seed=7
        )
        self.assertEqual(len(selected), 40)
        self.assertEqual(available["selected_paired_task_count"], 20)
        by_task = {}
        for item in selected:
            by_task.setdefault(item["task_id"], set()).add(item["reward_type"])
        self.assertTrue(
            all(values == {"gold_purchase", "wrong_purchase"} for values in by_task.values())
        )
    def test_target_ablation_uses_real_tool_names(self):
        variants = target_variants(
            {"asin": "123", "options": {"颜色": "黑色", "尺寸": "42"}}
        )
        self.assertIn("最终应购买商品", variants["prose_json"])
        calls = json.loads(variants["tool_sequence_json"])
        self.assertEqual(calls[0]["name"], "open_product")
        self.assertEqual(calls[-1], {"name": "buy_now", "arguments": {}})
        self.assertEqual([call["name"] for call in calls].count("select_option"), 2)

    def test_summary_enforces_directionality_gates(self):
        rows = []
        for outcome, utility, delta in (
            ("gold_purchase", 1.0, 1.0),
            ("wrong_purchase", -0.85, -1.0),
            ("valid_alternative_purchase", 0.55, 0.5),
        ):
            rows.append(
                {
                    "task_id": 9,
                    "reward_type": outcome,
                    "terminal_utility": utility,
                    "score_delta": delta,
                    "replay_valid": True,
                    "turns": [
                        {
                            "event": "open_gold_product",
                            "score_delta": delta,
                        }
                    ],
                }
            )
        summary = summarize_variant(rows)
        self.assertTrue(summary["passed"])
        chosen, passed = choose_variant({"good": summary})
        self.assertEqual(chosen, "good")
        self.assertTrue(passed)

    def test_phase0b_requires_significant_paired_direction(self):
        rows = []
        for task_id in range(20):
            for outcome, utility, delta in (
                ("gold_purchase", 1.0, 1.0 if task_id < 15 else -1.0),
                ("wrong_purchase", -0.85, 0.0),
            ):
                rows.append(
                    {
                        "task_id": task_id,
                        "reward_type": outcome,
                        "terminal_utility": utility,
                        "score_delta": delta,
                        "replay_valid": True,
                        "turns": [
                            {"event": "open_gold_product", "score_delta": delta}
                        ],
                    }
                )
        summary = summarize_paired_gold_wrong(rows, minimum_pairs=20)
        self.assertTrue(summary["passed"])
        self.assertLess(
            summary["phase0b"]["one_sided_exact_binomial_pvalue"], 0.05
        )

    def test_assemble_pairs_n_turns_with_n_plus_one_state_scores(self):
        selected = [
            {
                "trajectory_id": "trajectory-1",
                "task_id": 9,
                "global_step": 1,
                "uid": "uid-9",
                "rollout_index": 0,
                "reward_type": "gold_purchase",
                "reward": {"terminal_utility": 1.0},
            }
        ]
        private_target = {"asin": "123", "options": {}}
        replays = [
            {
                "private_target": private_target,
                "replay_valid": True,
                "replay_reward_type": "gold_purchase",
                "turns": [{"index": 0, "tool": "buy_now", "event": "buy_now"}],
            }
        ]
        jobs = [
            {
                "replay_index": 0,
                "variant": "prose_json",
                "state_index": 0,
                "mean_log_prob": -2.0,
                "prefix_truncated": False,
            },
            {
                "replay_index": 0,
                "variant": "prose_json",
                "state_index": 1,
                "mean_log_prob": -1.0,
                "prefix_truncated": False,
            },
            {
                "replay_index": 0,
                "variant": "tool_sequence_json",
                "state_index": 0,
                "mean_log_prob": -3.0,
                "prefix_truncated": False,
            },
            {
                "replay_index": 0,
                "variant": "tool_sequence_json",
                "state_index": 1,
                "mean_log_prob": -2.5,
                "prefix_truncated": False,
            },
        ]
        rows = assemble_rows(selected, replays, jobs)
        self.assertEqual(rows["prose_json"][0]["turns"][0]["score_delta"], 1.0)
        self.assertEqual(rows["tool_sequence_json"][0]["score_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
