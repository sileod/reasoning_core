from dataclasses import dataclass

import random

from reasoning_core.template import Task, Entry, Config, edict
from reasoning_core.utils import score_scalar

TASK_META = {'parent_source_id': None,
 'idea': 'ledger_reconciliation (draw 1 of 1)',
 'hypothesis': 'HV-069',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/ledger_reconciliation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3238464831,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class LedgerConfig(Config):
    n_accounts: int = 5
    n_ops: int = 8
    max_amount: int = 9

    def apply_difficulty(self, level):
        self.n_accounts = sround(self.n_accounts + level)
        self.n_ops = sround(self.n_ops + level * 2)
        self.max_amount = sround(self.max_amount + level)


def sround(x):
    return int(x)


def apply_ops(ops, accounts):
    bal = dict(accounts)
    unmatched = None
    for t, dest, amount in ops:
        if t == 'debit':
            bal[dest] = bal.get(dest, 0) + amount
        elif t == 'credit':
            if bal.get(dest, 0) >= amount:
                bal[dest] = bal.get(dest, 0) - amount
            else:
                unmatched = (t, dest, amount)
                break
        elif t == 'hold':
            bal[dest] = bal.get(dest, 0) + amount
            bal['__held__'] = bal.get('__held__', 0) + amount
        elif t == 'release':
            held = bal.get('__held__', 0)
            rel = min(amount, held)
            bal['__held__'] = held - rel
            if bal.get(dest, 0) >= amount:
                bal[dest] = bal.get(dest, 0) - amount
                bal['__held__'] = bal.get('__held__', 0) + amount
            elif held >= amount:
                bal['__held__'] = bal.get('__held__', 0) + amount
            else:
                unmatched = (t, dest, amount)
                break
        elif t == 'reversal':
            bal[dest] = bal.get(dest, 0) - amount
    return bal, unmatched


class LedgerReconciliation(Task):
    summary = ("Apply ordered debits, credits, holds, releases, reversals, and "
               "settlement rules across accounts, returning exact balances or "
               "an unmatched transaction with its failing reason.")
    config_cls = LedgerConfig

    def generate_entry(self):
        cfg = self.config
        names = [f"acct{i}" for i in range(cfg.n_accounts)]
        while True:
            init = {name: random.randint(0, cfg.max_amount) for name in names}
            ops = []
            for _ in range(cfg.n_ops):
                t = random.choice(['debit', 'credit', 'hold', 'release', 'reversal'])
                dest = random.choice(names)
                amount = random.randint(1, cfg.max_amount)
                ops.append((t, dest, amount))
            bal, unmatched = apply_ops(ops, init)
            bal.pop('__held__', None)
            if all(v >= 0 for v in bal.values()):
                break
        init_s = {k: v for k, v in sorted(init.items())}
        ops_s = [list(o) for o in ops]
        if unmatched is None:
            bal_s = {k: v for k, v in sorted(bal.items())}
            answer = ";".join(f"{k}:{v}" for k, v in sorted(bal_s.items()))
        else:
            answer = f"unmatched {unmatched[0]} {unmatched[1]} {unmatched[2]}"
        metadata = edict({
            "init": init_s,
            "ops": ops_s,
            "unmatched": unmatched,
        })
        metadata.payload = {
            "init": init_s,
            "ops": ops_s,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = ["Starting balances: " + ", ".join(
            f"{k}={v}" for k, v in sorted(metadata.payload["init"].items()))]
        lines.append("Ordered ledger operations (dest is the account, "
                     "debit adds to it, credit deducts if the account has "
                     "enough, hold reserves the amount against available "
                     "funds, release first frees held funds then deducts, "
                     "reversal subtracts the amount from the account):")
        for t, d, a in metadata.payload["ops"]:
            lines.append(f"  {t} {d} {a}")
        lines.append("Processing stops at the first operation that fails (a "
                     "credit or release whose amount cannot be covered). If "
                     "every operation succeeds, give the final balances as "
                     "account=value pairs joined by semicolons, sorted by "
                     "account name. If processing stops early, give the "
                     "unmatched operation as one line like 'unmatched credit "
                     "acct3 5'. State the answer alone.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        a = str(answer).strip()
        gold = entry.answer
        if a.replace(' ', '') == gold.replace(' ', ''):
            return 1.0
        return 0.0
