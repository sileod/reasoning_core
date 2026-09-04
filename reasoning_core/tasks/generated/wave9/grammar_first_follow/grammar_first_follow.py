import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'grammar_first_follow (draw 1 of 1)',
 'hypothesis': 'HV-059',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/grammar_first_follow',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 617014168,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

EPS = "e"
EOF = "$"


@dataclass
class GrammarFirstFollowConfig(Config):
    n_nonterm: int = 5
    n_term: int = 3
    prod_rate: int = 2
    alt_max: int = 4

    def apply_difficulty(self, level):
        self.n_nonterm = sround(self.n_nonterm + level)
        self.n_term = sround(self.n_term + (level > 1) + (level > 3))
        self.prod_rate = sround(self.prod_rate + (level > 1) + (level > 3))
        self.alt_max = sround(self.alt_max + (level > 2) + (level > 4))


def _compute(productions, names, terms):
    """Return (nullable_set, first_dict, follow_dict) via fixed-point closure."""
    nonterms = set(names)
    first = {t: {t} for t in terms}
    first[EPS] = {EPS}
    for N in names:
        first[N] = set()

    nullable = set()
    changed = True
    while changed:
        changed = False
        for N in names:
            if N in nullable:
                continue
            for alt in productions[N]:
                if all(s == EPS or s in nullable for s in alt):
                    nullable.add(N)
                    changed = True

    changed = True
    while changed:
        changed = False
        for N in names:
            for alt in productions[N]:
                acc = set()
                for s in alt:
                    acc |= (first[s] - {EPS})
                    if EPS not in first[s]:
                        break
                else:
                    acc.add(EPS)
                if not acc <= first[N]:
                    first[N] |= acc
                    changed = True

    follow = {N: set() for N in names}
    follow[names[0]].add(EOF)
    changed = True
    while changed:
        changed = False
        for A in names:
            for alt in productions[A]:
                for i, B in enumerate(alt):
                    if B not in nonterms:
                        continue
                    after = alt[i + 1:]
                    acc = follow[A] if not after else set()
                    for s in after:
                        acc |= (first[s] - {EPS})
                        if EPS not in first[s]:
                            break
                    else:
                        if after:
                            acc |= follow[A]
                    if not acc <= follow[B]:
                        follow[B] |= acc
                        changed = True
    return nullable, first, follow


def _norm(sym_set):
    return "empty" if not sym_set else " ".join(sorted(sym_set))


class GrammarFirstFollow(Task):
    summary = "Compute FIRST and FOLLOW sets for context-free grammars with nullable and mutually recursive productions, returning a requested canonical set."

    config_cls = GrammarFirstFollowConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_nonterm
        T = cfg.n_term
        names = [f"N{i}" for i in range(n)]
        terms = [chr(ord("a") + i) for i in range(T)]

        for _ in range(50):
            prods = {N: [] for N in names}
            for i, N in enumerate(names):
                alts = []
                n_alts = random.randint(1, cfg.prod_rate + 1)
                tries = 0
                while len(alts) < n_alts and tries < 40:
                    tries += 1
                    if random.random() < 0.3:
                        k = random.randint(1, min(cfg.alt_max, len(names)))
                        alt = tuple(random.sample(names, k))
                    else:
                        alt = tuple(random.choice(names + terms) for _ in range(random.randint(1, cfg.alt_max)))
                    if alt not in alts:
                        alts.append(alt)
                prods[N] = [list(a) for a in alts]

            chain_left = names[1:]
            if chain_left:
                half = chain_left[::2]
                base = []
                for k in range(len(half)):
                    base.append(half[k])
                    base.append(random.choice(terms))
                if len(prods[names[0]]) == 0 or random.random() < 0.5:
                    prods[names[0]].append(base)

            try:
                nullable, first, follow = _compute(prods, names, terms)
            except Exception:
                continue

            valid = True
            for N in names:
                if not first[N] or not follow[N]:
                    valid = False
                    break
            if not valid:
                continue

            N = random.choice(names)
            want_first = random.random() < 0.75
            if want_first:
                result = first[N]
                incl = EPS if EPS in result else None
                syms = sorted(x for x in result if x != EPS)
                if incl:
                    syms.append(incl)
                answer = _norm(set(syms))
                query = f"FIRST({N})"
                fmt = ('the canonical FIRST set: distinct symbols that can start a string '
                       f'derived from {N}, sorted, entries separated by spaces, using {EPS!r} '
                       'to denote the empty string epsilon')
            else:
                result = follow[N]
                syms = sorted(x for x in result if x != EOF)
                if EOF in result:
                    syms.append(EOF)
                answer = _norm(set(syms))
                query = f"FOLLOW({N})"
                fmt = ('the canonical FOLLOW set: distinct symbols that can immediately follow '
                       f'{N} in a sentential form, sorted, entries separated by spaces, using '
                       f'{EOF!r} to denote the end-of-input marker; an empty set is written {_norm(set())!r}')

            lines = []
            for Ni, alts in prods.items():
                for alt in alts:
                    altstr = " ".join(alt) if alt else EPS
                    lines.append(f"{Ni} -> {altstr}")
            payload = {
                "nonterminals": ", ".join(names),
                "terminals": ", ".join(terms),
                "start": names[0],
                "productions": "\n".join(lines),
            }
            metadata = edict({"payload": payload, "query": query, "fmt": fmt})
            return Entry(metadata=metadata, answer=answer)

        raise RuntimeError("failed to build grammar instance")

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return (f"{payload}\n\nCompute {metadata.query}.\n\nThe answer is {metadata.fmt}.")

    def score_answer(self, answer, entry):
        try:
            got = _norm(set(answer.split()) if answer.strip() else set())
        except Exception:
            return 0.0
        return float(got == _norm(set(entry.answer.split())))
