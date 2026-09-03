import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround


@dataclass
class PatternMatchExhaustivenessConfig(Config):
    k_min: int = 4
    k_max: int = 6
    n_clauses_min: int = 2
    n_clauses_max: int = 4
    covered_rate: float = 0.25

    def apply_difficulty(self, level):
        self.k_min = sround(self.k_min + level)
        self.k_max = sround(self.k_max + level)
        self.n_clauses_max = sround(self.n_clauses_max + level)


def _partition(items, n):
    items = list(items)
    if not items:
        return []
    n = min(n, len(items))
    random.shuffle(items)
    clauses = [[] for _ in range(n)]
    for idx, it in enumerate(items):
        clauses[idx % n].append(it)
    return clauses


class PatternMatchExhaustiveness(Task):
    summary = ("Given a finite algebraic datatype whose constructors are covered by "
               "pattern-match clauses, decide whether every constructor is covered and "
               "report the number of uncovered constructors as a counterexample count.")
    config_cls = PatternMatchExhaustivenessConfig

    def generate_entry(self):
        cfg = self.config
        k = random.randint(cfg.k_min, cfg.k_max)
        idxs = list(range(k))
        names = [f"c{i}" for i in idxs]
        covered = random.random() < cfg.covered_rate

        if covered:
            covered_idxs = idxs
            clauses = _partition(covered_idxs, random.randint(cfg.n_clauses_min, cfg.n_clauses_max))
            n_uncovered = 0
        else:
            n_uncovered = random.randint(1, k)
            uncovered_set = set(random.sample(idxs, n_uncovered))
            covered_idxs = [i for i in idxs if i not in uncovered_set]
            ncl = random.randint(cfg.n_clauses_min, cfg.n_clauses_max)
            clauses = _partition(covered_idxs, ncl) if covered_idxs else []
            assert n_uncovered >= 1 and n_uncovered <= k

        display = random.sample(names, k)
        clause_rows = []
        for clause in clauses:
            clause_rows.append(", ".join(names[i] for i in sorted(clause)))
        if not clause_rows:
            clause_rows = ["(no clause matches any constructor)"]

        answer = str(n_uncovered)

        metadata = edict(
            names=names,
            display_order=display,
            clauses=clause_rows,
        )
        metadata.payload = {
            "names": names,
            "display_order": display,
            "clauses": clause_rows,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        ctor_line = ", ".join(metadata.display_order)
        clause_lines = "\n".join(f"- clause {i+1}: covers {c}" for i, c in enumerate(metadata.clauses))
        return (
            f"Consider an algebraic datatype whose constructors are: {ctor_line}.\n"
            f"The following pattern-match clauses together cover constructors:\n"
            f"{clause_lines}\n"
            f"Determine how many constructors are NOT covered by any clause.\n"
            f'Answer with that number alone (for example "2"). '
            f"It is 0 exactly when every constructor is covered."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        if answer.strip() == entry.answer:
            return 1.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'pattern_match_exhaustiveness (draw 1 of 2)',
 'hypothesis': 'W1-060',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/pattern_match_exhaustiveness',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1344474452,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
