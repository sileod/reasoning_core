import math, random, re
from dataclasses import dataclass
from itertools import product

from gramforge import generate
from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround
from reasoning_core.utils import score_space_ints

from reasoning_core.tasks.probabilistic_reasoning import (
    qprobs,
    hidden_atoms,
    sorted_lits,
    evidence_grammar,
    boolean_value,
    influential_atoms,
    lit_options,
)


TASK_META = {'parent_source_id': '8d4361f034c05bb45bfb620d8d5ac97684b4e0e43755e9ebdb284cf053049b77',
 'idea': 'Test probability decision-boundary sampling.',
 'hypothesis': 'H8',
 'changes': 'Present two complete MAP explanations and control their '
            'log-probability margin.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2846187700,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 28,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def ranked_explanations(src):
    """Rank all complete assignments by posterior probability given the evidence."""
    atoms = hidden_atoms(src)
    queries, keys = [], []
    for i, bits in enumerate(product([False, True], repeat=len(atoms))):
        name = f"mpe_{i}"
        body = ", ".join(a if b else rf"\+{a}" for a, b in zip(atoms, bits))
        lits = [a if b else f"not {a}" for a, b in zip(atoms, bits)]
        queries += [f"{name} :- {body}.", f"query({name})."]
        keys.append((name, sorted_lits(lits)))
    p = qprobs(src + "\n" + "\n".join(queries))
    ranked = sorted(((p.get(k, 0.0), lits) for k, lits in keys), reverse=True)
    return [(prob, lits) for prob, lits in ranked if prob > 1e-12]


def direct_rank(atoms, probs, formula):
    """Fast posterior ranking of complete assignments via formula evaluation.

    posterior(assign | obs) is proportional to prod(p_i or (1-p_i)) whenever the
    assignment satisfies the observation formula, else zero. Normalization cancels
    in the ratio, so this yields the same top-two ordering as a full ProbLog run.
    """
    weights = []
    for bits in product([False, True], repeat=len(atoms)):
        values = dict(zip(atoms, bits))
        if not boolean_value(formula, values):
            continue
        w = 1.0
        for a in atoms:
            w *= probs[a] if values[a] else (1.0 - probs[a])
        lits = sorted_lits([a if b else f"not {a}" for a, b in values.items()])
        weights.append((w, lits))
    weights.sort(reverse=True)
    return [(w, lits) for w, lits in weights if w > 1e-12]


@dataclass
class NearTieMapConfig(Config):
    depth: int = 5
    min_atoms: int = 2
    max_atoms: int = 3
    max_formula_attempts: int = 120
    prob_trials: int = 300
    min_log_margin: float = 0.004
    max_log_margin: float = 0.45
    min_references: int = 3
    min_influential_atoms: int = 2
    min_shared_atoms: int = 0

    def apply_difficulty(self, level):
        self.depth = sround(self.depth + level)
        self.max_atoms = sround(min(6, self.max_atoms + level))
        self.min_atoms = sround(min(4, self.min_atoms + level // 2))
        self.max_log_margin = max(0.03, self.max_log_margin * (0.72 ** level))
        self.min_log_margin = max(0.002, self.min_log_margin * (0.8 ** level))
        self.min_references = sround(self.min_references + level)
        self.min_influential_atoms = sround(min(4, self.min_influential_atoms + level // 2))
        self.min_shared_atoms = sround(min(1, self.min_shared_atoms + level // 4))


class NearTieMap(Task):
    summary = "Choose the more probable of two near-tied complete explanations."
    def __init__(self, config=None):
        super().__init__(config=config or NearTieMapConfig())

    def generate_entry(self):
        cfg = self.config
        grid = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        for _ in range(cfg.max_formula_attempts):
            node = generate(evidence_grammar(), depth=cfg.depth, min_depth=4)
            formula = node @ "problog"
            formula_text = node @ "eng"
            references = re.findall(r"\b[a-f]\b", formula)
            atoms = sorted(set(references))
            if not cfg.min_atoms <= len(atoms) <= cfg.max_atoms:
                continue
            influential = influential_atoms(formula, atoms)
            shared = sum(references.count(a) > 1 for a in atoms)
            if (len(references) < cfg.min_references
                    or len(influential) < cfg.min_influential_atoms
                    or shared < cfg.min_shared_atoms):
                continue

            best = None
            for _ in range(cfg.prob_trials):
                probs = dict(zip(atoms, random.choices(grid, k=len(atoms))))
                ranked = direct_rank(atoms, probs, formula)
                if len(ranked) < 2:
                    continue
                (w1, hit), (w2, miss) = ranked[0], ranked[1]
                if hit == miss or w1 <= w2:
                    continue
                lm = math.log(w1 / w2)
                if cfg.min_log_margin <= lm <= cfg.max_log_margin:
                    best = (probs, hit, miss, lm)
                    break
            if best is None:
                continue
            probs, hit_lits, miss_lits, fast_lm = best

            src = "\n".join(
                [f"{p}::{a}." for a, p in probs.items()]
                + [f"observed :- {formula}.", "evidence(observed,true)."]
            )
            try:
                verified = ranked_explanations(src)
            except Exception:
                continue
            if len(verified) < 2:
                continue
            (p1, hit_lits), (p2, miss_lits) = verified[0], verified[1]
            if hit_lits == miss_lits or p1 <= p2:
                continue
            log_margin = math.log(p1 / p2)
            if not cfg.min_log_margin <= log_margin <= cfg.max_log_margin:
                continue

            opts = lit_options(src, shuffle_pairs=True)
            hit_idx = [opts.index(x) for x in hit_lits]

            def vals(lits):
                m = {}
                for l in lits:
                    m[l.removeprefix("not ")] = "false" if l.startswith("not ") else "true"
                return ", ".join(f"{a}={m[a]}" for a in atoms)

            lines = [f"Factor {a} is independently true with probability {p}."
                     for a, p in probs.items()]
            lines.append(f"The observation holds exactly when {formula_text}. We observe it.")
            lines.append("Hidden factor values:")
            for litn in (l for a in atoms for l in (a, f"not {a}")):
                lines.append(f"{opts.index(litn)}. {litn}")
            lines.append("Two complete explanations are under consideration:")
            lines.append(f"Explanation A: {vals(hit_lits)}")
            lines.append(f"Explanation B: {vals(miss_lits)}")
            lines.append("Which of these two complete explanations is more probable?")
            lines.append("Answer with the space-separated indexes (from the options above) "
                         "of the winning explanation's chosen values.")

            metadata = edict(
                problog=src,
                english="\n".join(lines),
                options=opts,
                n_atoms=len(atoms),
                probability_winner=str(p1),
                probability_runner=str(p2),
                log_margin=float(log_margin),
                winner_lits=hit_lits,
                runner_lits=miss_lits,
                probabilities=probs,
            )
            return Entry(metadata, " ".join(map(str, hit_idx)))
        raise RuntimeError("Failed to generate a near-tie MAP explanation task")

    def render_prompt(self, m):
        return m.english

    def score_answer(self, answer, entry):
        return score_space_ints(answer, entry)

    def balancing_key(self, problem):
        return "".join(problem.metadata.winner_lits)
