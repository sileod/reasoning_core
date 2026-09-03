import random
from dataclasses import dataclass
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add exact conditional-probability reasoning over a described finite '
         'experiment.',
 'hypothesis': 'S35',
 'changes': 'Ask for a conditional or posterior probability as a fraction in '
            'lowest terms.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1225289588,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _fmt(f):
    if f.denominator == 1:
        return str(f.numerator)
    return "%d/%d" % (f.numerator, f.denominator)


def _branch_wts(n):
    while True:
        ws = [random.randint(1, 9) for _ in range(n)]
        tot = sum(ws)
        out = [Fraction(w, tot) for w in ws]
        if all(0 <= w <= 1 for w in out):
            return out


def _parse_frac(s):
    import re
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(s))
    if not m:
        return None
    d = int(m.group(2))
    if d == 0:
        return None
    return Fraction(int(m.group(1)), d)


@dataclass
class ConditionalProbabilityConfig(Config):
    n_stages: int = 3
    n_branches: int = 3

    def apply_difficulty(self, level):
        self.n_stages = sround(self.n_stages + level)
        self.n_branches = min(5, sround(self.n_branches + (level // 2)))


COLOR_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


def _sum_over_paths(stages, keep_at_stage, keep_val, full_keep_val):
    """Sum the probability of every branch path.

    At keep_at_stage the drawn branch is pinned to keep_val. full_keep_val, when
    not None, pins every stage to that same branch. The finite outcome space is
    summed by a DP over stages that keeps a running total mass, which equals the
    exact finite-outcome sum without materializing every path.
    """
    total = Fraction(1)
    for s_idx in range(len(stages)):
        acc = Fraction(0)
        for b_idx, wt in enumerate(stages[s_idx]):
            if b_idx != keep_val and (keep_at_stage == s_idx or full_keep_val is not None):
                continue
            if full_keep_val is not None and b_idx != full_keep_val:
                continue
            acc += wt
        total *= acc
    return total


class ConditionalProbability(Task):
    config_cls = ConditionalProbabilityConfig

    def generate_entry(self):
        return self._try_gen()

    def _try_gen(self):
        for _ in range(50000):
            entry = self._try_once()
            if entry is not None:
                return entry
        raise RuntimeError("unable to draw a valid instance")

    def _try_once(self):
        n_stages = self.config.n_stages
        n_branches = self.config.n_branches
        if n_branches > len(COLOR_NAMES):
            n_branches = len(COLOR_NAMES)

        stages = []
        for _ in range(n_stages):
            wts = _branch_wts(n_branches)
            assert all(0 <= w <= 1 for w in wts)
            assert sum(wts) == 1
            stages.append(wts)

        cond_stage = random.randrange(n_stages)
        cond_branch = random.randrange(n_branches)

        den = _sum_over_paths(stages, cond_stage, cond_branch, None)
        if den == 0:
            return None
        num = _sum_over_paths(stages, cond_stage, cond_branch, cond_branch)
        ans = num / den
        if not (0 < ans < 1):
            return None

        story = []
        for s_idx in range(n_stages):
            wts = stages[s_idx]
            toks = []
            for b_idx, wt in enumerate(wts):
                toks.append("%s (probability %s)" % (COLOR_NAMES[b_idx], _fmt(wt)))
            story.append("At step %d a branch is drawn with these probabilities: %s."
                         % (s_idx + 1, "; ".join(toks)))

        cn = COLOR_NAMES[cond_branch]
        question = (
            "An experiment runs in %d steps. %s "
            "Given that the branch at step %d is %s, what is the conditional probability "
            "that the branch at every step is %s? Give your answer as a fraction in lowest terms."
            % (n_stages, " ".join(story), cond_stage + 1, cn, cn)
        )

        metadata = edict({
            "n_stages": n_stages,
            "n_branches": n_branches,
            "cond_stage": cond_stage,
            "cond_branch": cond_branch,
            "stages": [[str(w) for w in ws] for ws in stages],
            "payload": {"question": question},
        })
        answer = "%d/%d" % (ans.numerator, ans.denominator)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return render_payload(metadata.payload)

    def score_answer(self, answer, entry):
        parsed = _parse_frac(answer)
        if parsed is None:
            return 0.0
        gt = Fraction(entry.answer)
        if parsed == gt:
            return 1.0
        return 0.0
