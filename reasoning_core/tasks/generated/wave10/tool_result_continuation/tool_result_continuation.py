import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

STATES = ("success", "empty", "partial", "failure")


@dataclass
class ToolResultContinuationV2Config(Config):
    n_calls_min: int = 2
    n_calls_max: int = 3
    target_max: int = 12
    return_max: int = 6

    def apply_difficulty(self, level):
        self.n_calls_min = sround(2 + level)
        self.n_calls_max = sround(3 + 2 * level)
        self.target_max = sround(8 + 3 * level)
        self.return_max = sround(5 + level)


def _classify(total, target, has_failure):
    if has_failure:
        return "failure"
    if total >= target:
        return "success"
    if total == 0:
        return "empty"
    return "partial"


class ToolResultContinuation(Task):
    summary = ("Classify a sequence of tool call outcomes as success, empty, partial, or failure "
               "and report the accumulated item total that witnesses the state.")
    config_cls = ToolResultContinuationV2Config

    def generate_entry(self):
        c = self.config
        state = random.choice(STATES)

        if state == "empty":
            n = random.randint(c.n_calls_min, c.n_calls_max)
            calls = [("ok", 0) for _ in range(n)]
            target = random.randint(1, c.target_max)
        elif state == "success":
            n = random.randint(c.n_calls_min, c.n_calls_max)
            target = random.randint(1, c.target_max)
            returns = [random.randint(1, c.return_max) for _ in range(n)]
            while sum(returns) < target:
                returns[random.randrange(n)] += random.randint(1, c.return_max)
            calls = [("ok", r) for r in returns]
        elif state == "partial":
            n = random.randint(c.n_calls_min, c.n_calls_max)
            while True:
                returns = [random.randint(0, c.return_max) for _ in range(n)]
                if sum(returns) == 0:
                    continue
                lo = sum(returns) + 1
                hi = max(lo, c.target_max)
                target = random.randint(lo, hi)
                break
            calls = [("ok", r) for r in returns]
        else:  # failure
            n = random.randint(c.n_calls_min, c.n_calls_max)
            n_fail = random.randint(1, n)
            calls = [("fail", random.randint(1, 9)) for _ in range(n_fail)]
            calls += [("ok", random.randint(0, c.return_max)) for _ in range(n - n_fail)]
            random.shuffle(calls)
            target = random.randint(1, c.target_max)

        total = sum(r for status, r in calls if status == "ok")
        has_failure = any(status == "fail" for status, _ in calls)
        gold_state = _classify(total, target, has_failure)
        assert gold_state == state, (gold_state, state)
        assert isinstance(total, int) and total >= 0

        metadata = edict({
            "calls": [{"status": s, "value": v} for s, v in calls],
            "target": target,
            "total": total,
            "gold_state": gold_state,
        })
        metadata.payload = {"calls": metadata["calls"], "target": metadata["target"]}
        answer = f"{gold_state}:{total}"
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = []
        for i, call in enumerate(metadata.payload["calls"], start=1):
            if call["status"] == "ok":
                lines.append(f"- call {i}: accepted {call['value']} items")
            else:
                lines.append(f"- call {i}: errored with code {call['value']}")
        calls_block = "\n".join(lines)
        payload = {"calls": metadata.payload["calls"], "target": metadata.payload["target"],
                   "calls_block": calls_block}
        prompt = (
            f"A collection script needs at least {payload['target']} items and makes "
            f"{len(payload['calls'])} tool calls. Each call either accepts some items or errors.\n"
            f"\nCalls:\n{calls_block}\n\n"
            f"Sum the items from calls that accepted items to get the running total. A call that "
            f"errors makes the overall operation fail. Otherwise the operation succeeds if the "
            f"running total is at least {payload['target']}, is empty if the running total is 0, "
            f"and is partial otherwise.\n\n"
            f"Classify the outcome as success, empty, partial, or failure, and report the running "
            f"total.\n\n"
            f"The answer is exactly one line of the form STATE:TOTAL where STATE is one of "
            f"success, empty, partial, failure and TOTAL is the running total (a non-negative "
            f"integer)."
        )
        return f"{prompt}"


def _parse_answer(answer):
    if not isinstance(answer, str):
        return None
    parts = answer.strip().split(":")
    if len(parts) != 2:
        return None
    state = parts[0].strip().lower()
    if state not in STATES:
        return None
    try:
        total = int(float(parts[1].strip()))
    except ValueError:
        return None
    if total < 0:
        return None
    return state, total


def score_answer(self, answer, entry):
    got = _parse_answer(answer)
    gold = _parse_answer(entry.answer)
    if got is None or gold is None:
        return 0.0
    return 1.0 if got == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'tool_result_continuation (draw 2 of 2)',
 'hypothesis': 'ASTRA0-08',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/tool_result_continuation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2955773006,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
