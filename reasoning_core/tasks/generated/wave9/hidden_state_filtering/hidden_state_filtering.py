"""Executable exact sum-product (forward) filtering over finite hidden-state Markov models.

Given a discrete finite hidden Markov model -- initial distribution, row-stochastic
transition matrix, row-stochastic emission matrix and an observation sequence -- compute
the filtered (online posterior) probability of a query state at a query time via the
forward algorithm. Returns a decimal probability rounded to three digits.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


def _row_stochastic(rng, n):
    """A row-stochastic vector of length n drawn from the uniform simplex."""
    weights = [rng.random() + 1e-6 for _ in range(n)]
    total = float(sum(weights))
    return [w / total for w in weights]


def _to_chars(vals):
    return " ".join(f"{v:.3f}" for v in vals)


def _filter(init, trans, emit, obs):
    """Forward algorithm.

    Returns the list of normalized filtered beliefs: beliefs[t][s] =
    P(state_t = s | obs_0..obs_t). Also returns the unnormalized filter mass at each t.
    """
    K = len(init)
    alpha = [init[i] * emit[i][obs[0]] for i in range(K)]
    z = float(sum(alpha))
    beliefs = [[a / z for a in alpha]]
    for o in obs[1:]:
        nxt = [0.0] * K
        for i in range(K):
            acc = 0.0
            for j in range(K):
                acc += alpha[j] * trans[j][i]
            nxt[i] = acc * emit[i][o]
        z = float(sum(nxt))
        alpha = nxt
        beliefs.append([a / z for a in alpha])
    return beliefs


def _filter2(init, trans, emit, obs):
    """Independent forward implementation (different recurrence shape) for cross-checking."""
    K = len(init)
    beliefs = []
    pred = None
    for o in obs:
        if pred is None:
            un = [init[i] * emit[i][o] for i in range(K)]
        else:
            un = [emit[i][o] * sum(pred[j] * trans[j][i] for j in range(K)) for i in range(K)]
        z = sum(un)
        pred = [u / z for u in un]
        beliefs.append(pred)
    return beliefs


def _parse_prob(answer):
    try:
        val = float(str(answer).strip())
    except (TypeError, ValueError):
        return None
    return val


@dataclass
class HiddenStateFilteringConfig(Config):
    n_states: int = 2
    n_obs: int = 3
    horizon: int = 4

    def apply_difficulty(self, level):
        self.n_states = sround(2 + level)
        self.n_obs = sround(2 + level)
        self.horizon = sround(4 + 2 * level)


class HiddenStateFiltering(Task):
    summary = ("Execute exact sum-product filtering in finite hidden-state Markov models "
               "across observation sequences, returning a queried posterior state probability.")
    config_cls = HiddenStateFilteringConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        K = max(2, int(cfg.n_states))
        M = max(2, int(cfg.n_obs))
        T = max(2, int(cfg.horizon))

        while True:
            init = _row_stochastic(random, K)
            trans = [_row_stochastic(random, K) for _ in range(K)]
            emit = [_row_stochastic(random, M) for _ in range(K)]

            obs = [random.randrange(M) for _ in range(T)]

            beliefs1 = _filter(init, trans, emit, obs)
            beliefs2 = _filter2(init, trans, emit, obs)
            ok = all(
                abs(beliefs1[t][s] - beliefs2[t][s]) < 1e-9
                for t in range(T) for s in range(K)
            )
            if not ok:
                continue

            t = random.randrange(T)
            s = random.randrange(K)
            gold = beliefs1[t][s]
            if 0.0 <= gold <= 1.0:
                break

        gold_str = f"{round(gold, 3):.3f}"
        assert 0.0 <= gold <= 1.0, "posterior probability must lie in [0, 1]"
        assert abs(sum(beliefs1[t]) - 1.0) < 1e-9, "filtered belief must be a normalized distribution"

        metadata = edict({
            "n_states": K,
            "n_obs": M,
            "horizon": T,
            "init": _to_chars(init),
            "transition": [ _to_chars(row) for row in trans ],
            "emission": [ _to_chars(row) for row in emit ],
            "observations": _to_chars(obs),
            "query_time": t + 1,
            "query_state": s,
            "gold": gold,
        })
        metadata.payload = {
            "Hidden states": str(K),
            "Observation symbols": str(M),
            "Initial distribution over hidden states": metadata.init,
            "Transition matrix (row i gives P(next | state i))": "\n".join(metadata.transition),
            "Emission matrix (row i gives P(observation | state i))": "\n".join(metadata.emission),
            "Observed sequence (one symbol per time step)": metadata.observations,
            "Question": (
                f"Using exact sum-product (forward) filtering, what is the posterior probability "
                f"that the hidden state at time {t + 1} is state {s} (states numbered 0..{K - 1}, "
                f"times numbered 1..{T})? Give a decimal rounded to three digits after the point, "
                f"e.g. 0.347."
            ),
        }
        return Entry(metadata=metadata, answer=gold_str)

    def render_prompt(self, metadata):
        return f"{render_payload(metadata.payload)}\n\nThe answer is a decimal probability rounded to three digits."

    def score_answer(self, answer, entry):
        gold = entry.metadata.get("gold")
        if gold is None:
            return 0.0
        val = _parse_prob(answer)
        if val is None:
            return 0.0
        return 1.0 if abs(val - float(gold)) <= 1e-3 else 0.0

    def distractor_candidates(self, entry):
        md = entry.metadata
        K = md.n_states
        gold = float(md.gold)
        prior_candidates = [0.0, 1.0, 0.5]
        # a few other states' posteriors require recomputation; fall back to generic slips
        flawed = [
            round(1.0 - gold, 3),
            round(gold + 0.1, 3),
            round(gold - 0.1, 3),
            round(md.gold + 0.5, 3),
        ] + prior_candidates
        for v in flawed:
            if 0.0 <= v <= 1.0 and abs(v - gold) > 1e-3:
                yield f"{v:.3f}"


TASK_META = {'parent_source_id': None,
 'idea': 'hidden_state_filtering (draw 1 of 1)',
 'hypothesis': 'HV-002',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/hidden_state_filtering',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1475571465,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
