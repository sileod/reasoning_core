"""Compare finite diagnostic tests under an ambiguity score and pick the best one.

Given a set of candidate hypotheses with prior probabilities and a set of
possible test outcomes (each test partitions the hypotheses), we evaluate tests
by their *expected remaining ambiguity* after observing the outcome, using the
Shannon entropy of the posterior as a proxy for residual uncertainty.  The task
is to select the test with the lowest expected posterior entropy and return its
label (the tie broken deterministically by ascending label).

Only tests that can actually reduce uncertainty are offered: every candidate
partition is nontrivial (at least two outcome blocks) and the ambiguity score is
computed over the full outcome distribution.
"""

from dataclasses import dataclass, field
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload


@dataclass
class DiagnosticTestConfig(Config):
    n_hypotheses: int = 5
    n_tests: int = 4

    def apply_difficulty(self, level):
        self.n_hypotheses = self.n_hypotheses + level
        self.n_tests = self.n_tests + level


def _entropy(prob_list):
    import math

    h = 0.0
    total = sum(prob_list)
    for p in prob_list:
        if p > 0:
            q = p / total
            h -= q * math.log2(q)
    return h


def _expected_entropy(priors, partition):
    """Expected posterior entropy after running a test whose outcomes are the blocks."""
    outcome_probs = [0.0] * len(partition)
    for i, p in enumerate(priors):
        block = -1
        for b, group in enumerate(partition):
            if i in group:
                block = b
                break
        outcome_probs[block] += p
    expected = 0.0
    for b, op in enumerate(outcome_probs):
        if op <= 0:
            continue
        cond = [priors[i] / op for i in partition[b]]
        expected += op * _entropy(cond)
    return expected


def _random_partition(n, rng, min_blocks=2):
    """Return a list of blocks (lists of hypothesis indices) forming a partition of range(n)."""
    while True:
        blocks = [[] for _ in range(rng.randint(min_blocks, n))]
        for i in range(n):
            blocks[rng.randrange(len(blocks))].append(i)
        blocks = [sorted(b) for b in blocks if b]
        if len(blocks) >= min_blocks:
            return blocks


class DiagnosticTestSelection(Task):
    summary = (
        "Choose the diagnostic test with the lowest expected posterior entropy over a "
        "prior distribution on hypotheses, comparing tests by their nontrivial outcome "
        "partitions; return the winning test label with deterministic tie-breaking."
    )
    config_cls = DiagnosticTestConfig

    def generate_entry(self):
        cfg = self.config
        n_hyp = cfg.n_hypotheses
        n_tes = cfg.n_tests
        labels = [chr(ord("A") + i) for i in range(n_tes)]

        priors = [random.random() for _ in range(n_hyp)]
        total = sum(priors)
        priors = [p / total for p in priors]

        partitions = []
        scores = {}
        for t in range(n_tes):
            part = _random_partition(n_hyp, random, min_blocks=2)
            partitions.append(part)
            scores[labels[t]] = _expected_entropy(priors, part)

        best_score = min(scores.values())
        best_cands = sorted(label for label, s in scores.items() if abs(s - best_score) < 1e-12)
        answer = best_cands[0]

        payload = {"hypotheses": [round(p, 4) for p in priors], "tests": {}}
        test_txt = {}
        for t in range(n_tes):
            blocks = partitions[t]
            desc = " | ".join(",".join(str(i + 1) for i in b) for b in blocks)
            test_txt[labels[t]] = desc
        payload["tests"] = test_txt

        metadata = edict({"payload": payload, "priors": priors,
                          "partition_blocks": [list(map(list, p)) for p in partitions],
                          "scores": {k: float(v) for k, v in scores.items()},
                          "answer": answer})

        # verifier: recompute and confirm the chosen test is minimal under tie-break
        rec = {k: float(_expected_entropy(priors, p)) for k, p in zip(labels, partitions)}
        assert abs(rec[answer] - best_score) < 1e-9
        assert answer == min(labels, key=lambda k: (rec[k], k))
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        pl = metadata.payload
        lines = ["A diagnostic clinic must identify which of several hypotheses is true. Prior "
                 "probabilities for the hypotheses are given as a list."]
        lines.append("Hypothesis priors: " + ", ".join(str(x) for x in pl["hypotheses"]))
        lines.append("")
        lines.append("Each candidate test partitions the hypotheses into outcome blocks. A test "
                     "returns one outcome, and the true hypothesis lies in exactly one block. "
                     "Blocks are written as comma-separated hypothesis numbers grouped by "
                     "'|'." )
        lines.append("Candidate tests:")
        for label, desc in pl["tests"].items():
            lines.append(f"  Test {label}: {{{desc}}}")
        lines.append("")
        lines.append("The residual ambiguity of an outcome block is the Shannon entropy "
                     "(bits) of the normalized posterior over the hypotheses in that block. "
                     "The expected ambiguity of a test is the probability-weighted sum across "
                     "its outcome blocks. Choose the test with the LOWEST expected ambiguity; "
                     "on a tie, choose the test with the alphabetical-least label.")
        lines.append("")
        lines.append('The answer is a single letter, the label of the best test, e.g. "B".')
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        gold = entry.answer
        if isinstance(answer, str):
            a = answer.strip().strip('"').strip("'")
            if a == gold:
                return 1.0
            if len(a) == 1 and a in entry.metadata.get("scores", {}):
                return 0.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'diagnostic_test_selection (draw 1 of 1)',
 'hypothesis': 'HV-007',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/diagnostic_test_selection',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1749059983,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
