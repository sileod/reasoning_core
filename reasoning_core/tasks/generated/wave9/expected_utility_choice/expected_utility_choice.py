from dataclasses import dataclass
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'expected_utility_choice (draw 1 of 1)',
 'hypothesis': 'HV-006',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/expected_utility_choice',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 798610012,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ExpectedUtilityConfig(Config):
    n_actions: int = 3
    n_outcomes: int = 3

    def apply_difficulty(self, level):
        self.n_actions = sround(self.n_actions + level)
        self.n_outcomes = sround(self.n_outcomes + level)


def _distribute_probabilities(n_outcomes, rng):
    cuts = sorted(rng.random() for _ in range(n_outcomes - 1))
    pts = [0.0] + cuts + [1.0]
    probs = [pts[i + 1] - pts[i] for i in range(n_outcomes)]
    return probs


def _canonical(answer):
    return "{:.6f}".format(answer[0])


def _parse_canonical(text):
    if not text:
        return []
    try:
        return [float(text.strip())]
    except Exception:
        return []


def _expected_utility(probs, payoffs):
    return sum(p * v for p, v in zip(probs, payoffs))


def _best_action(utility):
    maxu = max(utility)
    return utility.index(maxu)


class ExpectedUtilityChoice(Task):
    summary = ("Evaluate a lottery over outcomes with signed utilities against a "
               "reference payoff, return the maximizing action's exact expected "
               "utility, or a mix of exact EU and argmax modes.")
    config_cls = ExpectedUtilityConfig

    def generate_entry(self):
        cfg = self.config
        n_actions = cfg.n_actions
        n_outcomes = cfg.n_outcomes

        probs = [
            [round(p, 4) for p in _distribute_probabilities(n_outcomes, random)]
            for _ in range(n_actions)
        ]
        payoffs = []
        for _ in range(n_actions):
            utilities = [random.randint(-50, 100) for _ in range(n_outcomes)]
            payoffs.append(utilities)

        utilities = [_expected_utility(probs[i], payoffs[i]) for i in range(n_actions)]

        idx = _best_action(utilities)
        answer = _canonical([utilities[idx]])

        metadata = edict({
            "n_actions": n_actions,
            "n_outcomes": n_outcomes,
            "probs_projections": [[round(p, 4) for p in pr] for pr in probs],
            "payoffs": [[int(v) for v in row] for row in payoffs],
            "answer_float": float(round(utilities[idx], 6)),
            "payload": {
                "actions": [
                    {
                        "outcomes": " | ".join(
                            f"{p:.4f} -> {v}" for p, v in zip(probs[i], payoffs[i])
                        )
                    }
                    for i in range(n_actions)
                ]
            },
        })
        metadata.answer_float = float(round(utilities[idx], 6))

        entry = Entry(metadata=metadata, answer=answer)
        self._verify(entry)
        return entry

    def _verify(self, entry):
        metadata = entry.metadata
        probs = metadata["probs_projections"]
        payoffs = metadata["payoffs"]
        n_actions = len(payoffs)
        utilities = [
            sum(p * v for p, v in zip(probs[i], payoffs[i]))
            for i in range(n_actions)
        ]
        idx = _best_action(utilities)
        parsed = _parse_canonical(entry.answer)
        if len(parsed) == 1:
            assert abs(parsed[0] - utilities[idx]) < 1e-4
        else:
            raise RuntimeError("unexpected answer length")

    def render_prompt(self, metadata):
        lines = [
            "Each action is a lottery whose outcomes are signed payoffs (negative means a loss).",
        ]
        for i, act in enumerate(metadata.payload.actions):
            lines.append(f"Action {i}: " + act["outcomes"])
        lines.append(
            "The expected utility of an action is the sum of probability times payoff "
            "over its outcomes. Choose the action with the strictly greatest expected "
            "utility, breaking ties by the smallest action index, and report that "
            "action's exact expected utility (as a decimal)."
        )
        lines.append("\nThe answer is a single decimal number.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        parsed = _parse_canonical(str(answer).strip())
        if len(parsed) != 1:
            return 0.0
        true = entry.metadata.get("answer_float", float("nan"))
        return 1.0 if abs(parsed[0] - true) < 1e-4 else 0.0
