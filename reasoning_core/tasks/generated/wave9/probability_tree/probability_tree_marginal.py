from dataclasses import dataclass
import random
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround
from reasoning_core.utils import score_scalar

TASK_META = {'parent_source_id': None,
 'idea': 'probability_tree_marginal (draw 1 of 1)',
 'hypothesis': 'HV-004',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/probability_tree_marginal',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3536382515,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _fmt_frac(f):
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _parse_frac(s):
    s = s.strip()
    if '/' in s:
        num, den = s.split('/')
        return Fraction(int(num.strip()), int(den.strip()))
    return Fraction(int(s), 1)


@dataclass
class ProbabilityTreeConfig(Config):
    steps: int = 3
    branches: int = 2
    denom: int = 20

    def apply_difficulty(self, level):
        self.steps = sround(self.steps + level)
        self.branches = sround(self.branches + (level // 2))
        self.denom = 20 + 5 * level


class ProbabilityTree(Task):
    summary = ("Evaluate branching sequential random processes with conditional transitions, "
               "replacement or state changes, returning an exact event probability.")

    config_cls = ProbabilityTreeConfig

    def generate_entry(self):
        cfg = self.config
        steps = cfg.steps
        branches = cfg.branches
        if steps < 2:
            steps = 2
        if branches < 2:
            branches = 2
        denom = cfg.denom
        total = denom

        target_state = random.randrange(branches)

        def composition(k, total_sum):
            if k <= 1:
                return [total_sum]
            cuts = sorted(random.sample(range(total_sum + k - 2), k - 1))
            bars = [-1] + cuts + [total_sum + k - 2]
            return [bars[i + 1] - bars[i] - 1 for i in range(len(bars) - 1)]

        while True:
            probs = []
            ok = True
            for _ in range(steps):
                row = composition(branches, total)
                if row[target_state] < 1:
                    ok = False
                    break
                probs.append([Fraction(row[b], total) for b in range(branches)])
            if ok:
                break

        ans = Fraction(1)
        for step in range(steps):
            ans *= probs[step][target_state]

        if not (0 < ans <= 1):
            raise RuntimeError("probability out of domain")

        trans = []
        for step in range(steps):
            row = [_fmt_frac(probs[step][b]) for b in range(branches)]
            trans.append(row)

        metadata = edict({
            "steps": steps,
            "branches": branches,
            "target": target_state,
            "transitions": trans,
            "target_name": f"outcome {target_state+1}",
        })
        metadata.payload = {
            "process": {
                "steps": steps,
                "branches": branches,
                "target": target_state + 1,
            },
            "transitions": trans,
        }
        answer = _fmt_frac(ans)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        steps = metadata.steps
        branches = metadata.branches
        target = metadata.target + 1
        rows = []
        for s, row in enumerate(metadata.transitions):
            parts = []
            for b in range(branches):
                parts.append(f"outcome {b+1} with probability {row[b]}")
            rows.append(f"step {s+1}: " + ", ".join(parts))
        header = (f"A random process runs for {steps} steps. At each step it produces one of "
                  f"{branches} outcomes. The transition probabilities are:")
        lines = [header] + rows
        body = "\n".join(lines)
        prompt = (f"{body}\n\n"
                  f"At every step, outcome {target} has the chance shown above at that step. "
                  f"Compute the probability that the process produces outcome {target} at "
                  f"EVERY step, expressed as a fraction in lowest terms (e.g. 3/4 or 1/2). "
                  f"The answer is that fraction.")
        return prompt

    def score_answer(self, answer, entry):
        try:
            a = _parse_frac(answer)
        except Exception:
            return 0.0
        gold = _parse_frac(entry.answer)
        if a == gold:
            return 1.0
        return 0.0

    def distractor_candidates(self, entry):
        meta = entry.metadata
        steps = int(meta.steps)
        target = int(meta.target)
        transitions = meta.transitions
        probs = []
        for row in transitions:
            probs.append([_parse_frac(x) for x in row])
        candidates = []
        any_outcome = Fraction(1)
        for step in range(steps):
            s = sum(probs[step])
            any_outcome *= (s - probs[step][target])
        candidates.append(_fmt_frac(any_outcome))
        last_only = probs[steps - 1][target]
        candidates.append(_fmt_frac(last_only))
        for step in range(steps):
            candidates.append(_fmt_frac(probs[step][target]))
        return candidates
