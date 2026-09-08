"""After a fact is corrected, return only the previously answered items whose answers now change."""

from dataclasses import dataclass
import random

from reasoning_core.template import Config, edict, render_payload, stochastic_rounding as sround
from reasoning_core.template import Task, Entry


TASK_META = {'parent_source_id': None,
 'idea': 'answer_relevant_change (draw 1 of 2)',
 'hypothesis': 'ASTRA0-18',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/answer_relevant_change',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3783103188,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class AnswerRelevantChangeConfig(Config):
    n_items: int = 5
    n_corrections: int = 1
    value_range: int = 9

    def apply_difficulty(self, level):
        self.n_items = sround(self.n_items + level)
        self.n_corrections = max(1, sround(self.n_corrections + level // 2))
        self.value_range = sround(self.value_range + level)


class AnswerRelevantChange(Task):
    summary = ("Given a ranked list of labeled answer-items and one corrected fact, return only "
               "the labels whose rank order changes after the correction; labels unchanged in "
               "relative order are omitted.")
    config_cls = AnswerRelevantChangeConfig

    def generate_entry(self):
        while True:
            cfg = self.config
            n = cfg.n_items
            rng_values = [random.randint(0, cfg.value_range) for _ in range(n)]
            labels = list(range(1, n + 1))
            random.shuffle(labels)
            items = list(zip(labels, rng_values))

            # Rank each item by (value, label) descending.
            def rank(items_list):
                return sorted(items_list, key=lambda p: (-p[1], p[0]))

            # Choose an item (or subset) whose value will be corrected.
            correct_idx = random.randrange(n)
            old_val = items[correct_idx][1]
            base_rank = rank(items)
            pos_old = base_rank.index(items[correct_idx])

            # New value: must be different and change rank; try a bounded set of candidates.
            candidates = list(range(0, cfg.value_range + 2))
            candidates.remove(old_val)
            random.shuffle(candidates)
            changed_items = None
            new_val = None
            for cand in candidates:
                new_items = items.copy()
                new_items[correct_idx] = (items[correct_idx][0], cand)
                new_rank = rank(new_items)
                # Determine which labels flip relative order.
                old_order = [lab for lab, _ in base_rank]
                new_order = [lab for lab, _ in new_rank]
                changed = self._flips(old_order, new_order)
                if changed:
                    changed_items = changed
                    new_val = cand
                    break
            if changed_items is None:
                continue

            metadata = edict({
                "items": [{"label": f"item {lab}", "value": val} for lab, val in items],
                "ranked": [lab for lab, _ in base_rank],
                "correction_index": correct_idx,
                "old_value": old_val,
                "new_value": new_val,
                "correction_label": f"item {items[correct_idx][0]}",
            })
            sorted_items = sorted(items)
            metadata.payload = {
                "entries": sorted([(lab, val) for lab, val in items]),
                "ranking_rule": "larger value ranks higher; equal values break by smaller label",
                "corrected": f"item {items[correct_idx][0]} is changed from {old_val} to {new_val}",
            }

            # Verify: recompute the flips ourselves.
            new_items = items.copy()
            new_items[correct_idx] = (items[correct_idx][0], new_val)
            expected = self._flips([lab for lab, _ in base_rank],
                                   [lab for lab, _ in rank(new_items)])
            assert expected == changed_items, (expected, changed_items)
            if not changed_items:
                continue
            answer = ",".join(str(lab) for lab in sorted(changed_items))
            return Entry(metadata=metadata, answer=answer)

    @staticmethod
    def _flips(old_order, new_order):
        """Labels whose relative position among the others changes."""
        changed = set()
        n = len(old_order)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = old_order[i], old_order[j]
                npi = new_order.index(p1)
                npj = new_order.index(p2)
                if (npi < npj) != (i < j):
                    changed.add(p1)
                    changed.add(p2)
        return changed

    def render_prompt(self, metadata):
        payload = metadata.payload
        item_lines = "\n".join(
            f"- {lab}: {val}" for lab, val in payload["entries"]
        )
        return (
            f"{item_lines}\n"
            f"Ranking rule: {payload['ranking_rule']}.\n"
            f"You ranked every item above by this rule (larger value first, ties by smaller label). "
            f"Then this fact is corrected: {payload['corrected']}.\n\n"
            f"Keeping every other item unchanged, which of your previously ranked items now sit in "
            f"a different relative rank than before? Give the answer as the labels of exactly the "
            f"items whose rank order changed, comma-separated in ascending label order. Items whose "
            f"relative order is unchanged are omitted. Write only the comma-separated list, nothing "
            f"else."
        )

    def score_answer(self, answer, entry):
        try:
            answer = answer.strip().strip("[]").replace(" ", "")
            if answer == "":
                return 0.0
            given = {int(x) for x in answer.split(",") if x != ""}
        except Exception:
            return 0.0
        expected = {int(x) for x in entry.answer.split(",")}
        if given == expected:
            return 1.0
        if given and given < expected:
            return 0.0
        return 0.0
