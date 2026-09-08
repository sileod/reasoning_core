"""Fulfil a request after successive corrections, cancellations, and reinstatements."""

import random

from reasoning_core.template import Config, Entry, Task, edict


class RequestRevisionConfig(Config):
    n_ops: int = 6

    def apply_difficulty(self, level):
        self.n_ops = 4 + level * 2


class RequestRevision(Task):
    summary = (
        "Fulfil a request after successive corrections, cancellations, and "
        "reinstatements; preserve requirements that were not revised."
    )
    config_cls = RequestRevisionConfig

    def generate_entry(self):
        cfg = self.config
        max_lines = 12
        lines = list(range(1, max_lines + 1))

        start = set(random.sample(lines, k=random.randint(1, 3)))

        corrections = []  # each: {"idx", "kind", "line", "active"}
        decision_log = []  # human-visible ops: correction/cancel/reinstate records
        next_idx = 0

        for _ in range(cfg.n_ops):
            can_cancel = any(c["active"] for c in corrections)
            can_reinstate = any(not c["active"] for c in corrections)
            pool = ["add", "remove"]
            if can_cancel:
                pool.append("cancel")
            if can_reinstate:
                pool.append("reinstate")
            kind = random.choice(pool)

            if kind in ("add", "remove"):
                existing = set(c["line"] for c in corrections if c["active"])
                if kind == "add":
                    choices = [l for l in lines if l not in (start | existing)]
                    if not choices:
                        kind = "remove"
                if kind == "remove":
                    removable = (start | existing) - set(
                        c["line"] for c in corrections if c["kind"] == "add" and not c["active"]
                    )
                    choices = sorted(removable)
                    if not choices:
                        kind = "add"
                        choices = [l for l in lines if l not in (start | existing)]
                if kind == "add":
                    line = random.choice(choices)
                else:
                    line = random.choice(choices)
                corrections.append({"idx": next_idx, "kind": kind, "line": line, "active": True})
                decision_log.append(("correction", next_idx, kind, line))
                next_idx += 1
            elif kind == "cancel":
                pool_c = [c["idx"] for c in corrections if c["active"]]
                idx = random.choice(pool_c)
                for c in corrections:
                    if c["idx"] == idx:
                        c["active"] = False
                decision_log.append(("cancel", idx))
            elif kind == "reinstate":
                pool_r = [c["idx"] for c in corrections if not c["active"]]
                idx = random.choice(pool_r)
                for c in corrections:
                    if c["idx"] == idx:
                        c["active"] = True
                decision_log.append(("reinstate", idx))

        active = set(start)
        for c in corrections:
            if c["active"]:
                if c["kind"] == "add":
                    active.add(c["line"])
                else:
                    active.discard(c["line"])
        final_lines = sorted(active)

        verify = set(start)
        for c in corrections:
            if c["active"]:
                if c["kind"] == "add":
                    verify.add(c["line"])
                else:
                    verify.discard(c["line"])
        assert sorted(verify) == final_lines

        answer = ",".join(str(l) for l in final_lines) if final_lines else "none"
        assert answer == "none" or all(0 < int(x) <= max_lines for x in answer.split(","))

        metadata = edict()
        metadata.start = sorted(start)
        metadata.final = final_lines
        metadata.payload = {
            "start": metadata.start,
            "steps": self._render_steps(decision_log),
        }
        return Entry(metadata=metadata, answer=answer)

    def _render_steps(self, log):
        out = []
        for i, item in enumerate(log):
            kind = item[0]
            if kind == "correction":
                _, idx, k, line = item
                out.append(
                    f"Instruction {i+1}: {'add' if k == 'add' else 'remove'} line {line} "
                    f"(requirement R{idx})."
                )
            elif kind == "cancel":
                _, idx = item
                out.append(f"Instruction {i+1}: cancel requirement R{idx}.")
            else:
                _, idx = item
                out.append(f"Instruction {i+1}: reinstate requirement R{idx}.")
        return out

    def render_prompt(self, metadata):
        lines = ", ".join(str(l) for l in metadata.start)
        body = [f"Lines {lines} are in force initially."]
        body.extend(metadata.payload["steps"])
        body.append("After all instructions, which lines remain in force? Give the answer as a comma-separated list, or 'none' if no line remains.")
        return "\n".join(body)

    def score_answer(self, answer, entry):
        gold = list(entry.metadata.final)
        if gold:
            try:
                s = str(answer).strip()
                if not s:
                    return 0.0
                got = [int(x) for x in s.replace(" ", "").split(",") if x]
            except Exception:
                return 0.0
            return 1.0 if sorted(got) == gold else 0.0
        else:
            try:
                return 1.0 if str(answer).strip().lower() == "none" else 0.0
            except Exception:
                return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'request_revision (draw 1 of 2)',
 'hypothesis': 'ASTRA0-02',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/request_revision',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3926271670,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
