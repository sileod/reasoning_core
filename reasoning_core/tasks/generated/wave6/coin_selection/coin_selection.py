"""Coin selection: fewest coins to make an amount where greedy is wrong.

The task asks for the fewest coins (unbounded-change DP, never greedy) that sum
exactly to an amount, returned as a comma-separated list of coin values in
decreasing order, repeating a value as often as it is used. Denominations are
chosen so that the greedy answer is NOT optimal for the amount asked about --
that is what makes the task non-trivial.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround


@dataclass
class CoinSelectionConfig(Config):
    n_denoms: int = 5
    max_amount: int = 60

    def apply_difficulty(self, level):
        self.n_denoms = sround(self.n_denoms + 2 * level)
        self.max_amount = sround(self.max_amount + 25 * level)


def _unbounded_min_count(denoms, amount):
    """Min number of coins to make amount with unlimited denoms, or None."""
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        best = INF
        for d in denoms:
            if d <= a and dp[a - d] + 1 < best:
                best = dp[a - d] + 1
        dp[a] = best
    return dp[amount] if dp[amount] != INF else None


def _greedy_count(denoms, amount):
    """Number of coins using the greedy (largest-first) strategy."""
    d_sorted = sorted(denoms, reverse=True)
    remaining = amount
    count = 0
    for d in d_sorted:
        while remaining >= d:
            remaining -= d
            count += 1
    if remaining != 0:
        return None
    return count


def _min_count_under(amount, denoms, cap):
    """Min coins to make amount using only denominations <= cap; 1 is always usable."""
    denoms_u = [d for d in denoms if d <= cap]
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        best = INF
        for d in denoms_u:
            if d <= a and dp[a - d] + 1 < best:
                best = dp[a - d] + 1
        dp[a] = best
    return dp[amount] if dp[amount] != INF else None


def _lex_largest_min_multiset(denoms, amount):
    """Min-coin multiset (decreasing value list); lexicographically largest.

    Picks the largest possible coin at each position that still admits an
    optimal completion, which yields the lex-largest among minimum-count
    multisets.
    """
    opt = _unbounded_min_count(denoms, amount)
    if opt is None:
        return None
    denoms_u = sorted(set(denoms), reverse=True)
    result = []
    remaining = amount
    cap = denoms_u[0]
    need = opt
    while remaining > 0:
        placed = False
        for c in denoms_u:
            if c > cap or c > remaining:
                continue
            nxt = remaining - c
            if nxt == 0:
                if need == 1:
                    result.append(c)
                    remaining = 0
                    placed = True
                    break
                continue
            if _min_count_under(nxt, denoms_u, c) == need - 1:
                result.append(c)
                remaining = nxt
                cap = c
                need -= 1
                placed = True
                break
        if not placed:
            return None
    return result


def _greedy_fails(denoms, amount):
    opt = _unbounded_min_count(denoms, amount)
    if opt is None:
        return False
    greedy = _greedy_count(denoms, amount)
    if greedy is None:
        return False
    return greedy > opt


class CoinSelection(Task):
    """Fewest coins to make an amount where greedy change-making fails."""

    config_cls = CoinSelectionConfig

    def generate_entry(self):
        cfg = self.config
        while True:
            denoms = sorted(
                random.sample(range(2, cfg.max_amount), cfg.n_denoms), reverse=True
            )
            denoms_plus1 = sorted(set(denoms + [1]), reverse=True)
            amount = random.randint(30, cfg.max_amount)
            if not _greedy_fails(denoms_plus1, amount):
                continue
            best = _lex_largest_min_multiset(denoms_plus1, amount)
            if best is None:
                continue
            assert sum(best) == amount
            assert _unbounded_min_count(denoms_plus1, amount) == len(best)
            greedy = _greedy_count(denoms_plus1, amount)
            assert greedy is not None and greedy > len(best)
            break

        metadata = edict(
            {
                "denominations": [int(x) for x in denoms_plus1],
                "amount": int(amount),
                "opt_count": int(len(best)),
                "greedy_count": int(greedy),
            }
        )
        metadata.payload = {
            "denominations": [int(x) for x in denoms_plus1],
            "amount": int(amount),
        }
        answer = ", ".join(str(int(x)) for x in best)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        denoms = ", ".join(str(int(x)) for x in metadata.payload["denominations"])
        amount = int(metadata.payload["amount"])
        return (
            f"A till has coins of denominations {denoms}. It must make exactly "
            f"{amount} cents using the fewest coins possible, with any number of "
            f"each denomination available. When several combinations tie at the "
            f"fewest coins, pick the one whose list is lexicographically largest. "
            f"What is the answer? Give a comma-separated list of coin values in "
            f"decreasing order, repeating a value as often as it is used."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        try:
            got = [int(t.strip()) for t in str(answer).split(",") if t.strip() != ""]
        except (ValueError, TypeError):
            return 0.0
        if not got:
            return 0.0
        gold = [int(x) for x in entry.answer.split(",")]
        return 1.0 if got == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'Fewest coins for an amount in a currency where greedy is wrong.',
 'hypothesis': 'S64',
 'changes': 'New task; the answer is the coin multiset, and greedy fails on '
            'it.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 241888686,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
