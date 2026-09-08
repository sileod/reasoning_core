"""Choose the observation/question that best splits a finite set of hypotheses.

Instance family: a finite set of candidate hypotheses, each a fixed-length string
over a small alphabet. A question reveals the character at one position (0-based
index) of the unknown true hypothesis. The best question is the one whose
response partitions the remaining hypotheses into the greatest number of distinct
groups; ties go to the smallest position index.
"""

import random as _random

from reasoning_core.template import Task, Entry, Config, edict, render_payload

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _group_counts(hypotheses, pos):
    return len({h[pos] for h in hypotheses})


def _best_position(hypotheses, m):
    best_p = None
    best_d = -1
    for p in range(m):
        d = _group_counts(hypotheses, p)
        if d > best_d or (d == best_d and (best_p is None or p < best_p)):
            if d > best_d:
                best_d = d
                best_p = p
        elif d == best_d and p < best_p:
            best_p = p
            best_d = d
    return best_p, best_d


class InfoSeekingConfig(Config):
    n_hypos: int = 8
    m_len: int = 7

    def apply_difficulty(self, level):
        self.n_hypos = 6 + 2 * level
        self.m_len = 5 + level


class InfoSeeking(Task):
    summary = ("Choose the observation or question that best distinguishes remaining "
               "hypotheses under a supplied finite model, scoring by resulting group "
               "split size with smallest-index tie-break.")
    config_cls = InfoSeekingConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_hypos
        m = cfg.m_len

        while True:
            hypos = [
                "".join(_random.choice(_ALPHABET) for _ in range(m))
                for _ in range(n)
            ]
            # At least two distinct strings so there is something to distinguish.
            if len(set(hypos)) >= 2:
                break

        p_best, d_best = _best_position(hypos, m)

        for p in range(m):
            d = _group_counts(hypos, p)
            assert d <= d_best
            if p < p_best:
                assert d < d_best
            elif p == p_best:
                assert d == d_best

        answer = f"{p_best}:{d_best}"
        hypos_sorted = sorted(hypos)
        metadata = edict({
            "hypos": hypos_sorted,
            "m": m,
            "alphabet": _ALPHABET,
            "p": p_best,
            "d": d_best,
        })
        metadata.payload = {
            "hypos": hypos_sorted,
            "m": m,
            "alphabet": _ALPHABET,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = "\n".join(f"  {h}" for h in metadata.payload["hypos"])
        payload = render_payload(metadata.payload)
        return (
            f"{payload}\n\n"
            "Candidate hypotheses are the strings above, each a length-"
            f"{metadata.payload['m']} string over the alphabet "
            f"{metadata.payload['alphabet']}. The true hypothesis is one of them.\n"
            "A question reveals the character of the true hypothesis at exactly one "
            "position (0-based index). The best question is the one whose response "
            "partitions the remaining hypotheses into the most distinct groups; if "
            "several positions give the maximum number of groups, the best is the "
            "smallest such position index.\n\n"
            "Which single position should we ask to maximize the number of groups?\n"
            "Answer as p:d where p is that position index and d is the number of "
            "distinct groups the question produces."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        ans = answer.strip()
        gold = entry.answer
        if ans == gold:
            return 1.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'information_seeking (draw 1 of 2)',
 'hypothesis': 'ASTRA0-19',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/information_seeking',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 147658999,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
