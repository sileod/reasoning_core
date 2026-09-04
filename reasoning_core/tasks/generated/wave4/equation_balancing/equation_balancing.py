import random
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

from reasoning_core.template import Config, Entry, Task, edict

TASK_META = {'parent_source_id': None,
 'idea': 'Add chemical equation balancing as exact integer linear algebra.',
 'hypothesis': 'S45',
 'changes': 'Ask for the smallest whole-number coefficients that balance a '
            'stated reaction.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 16602065,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

ELEMENTS = ("H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
            "P", "S", "Cl", "Ar", "K", "Ca", "Fe", "Cu", "Zn", "Br", "Ag", "I", "Ba", "Au")


def _lcm(a, b):
    return a * b // gcd(a, b)


def _solve(matrix, n_cols):
    mat = [[Fraction(x) for x in row] for row in matrix]
    n_rows = len(mat)
    rank = 0
    pivots = []
    for col in range(n_cols):
        pivot = None
        for r in range(rank, n_rows):
            if mat[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        pv = mat[rank][col]
        mat[rank] = [x / pv for x in mat[rank]]
        for r in range(n_rows):
            if r != rank and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[rank])]
        pivots.append(col)
        rank += 1
    if rank != n_cols - 1:
        return None
    free = [c for c in range(n_cols) if c not in pivots]
    vec = [Fraction(0) for _ in range(n_cols)]
    vec[free[0]] = Fraction(1)
    for col in pivots:
        row = pivots.index(col)
        vec[col] = -mat[row][free[0]]
    denom = 1
    for f in vec:
        denom = _lcm(denom, f.denominator)
    intvec = [int(f * denom) for f in vec]
    g = 0
    for v in intvec:
        g = gcd(g, abs(v))
    if g == 0:
        return None
    intvec = [v // g for v in intvec]
    if any(v <= 0 for v in intvec):
        return None
    return intvec


def _parse_formula(formula):
    counts = {}
    i = 0
    while i < len(formula):
        if not formula[i].isupper():
            return None
        j = i + 1
        while j < len(formula) and formula[j].islower():
            j += 1
        sym = formula[i:j]
        k = j
        while k < len(formula) and formula[k].isdigit():
            k += 1
        num = int(formula[j:k]) if k > j else 1
        counts[sym] = counts.get(sym, 0) + num
        i = k
    return counts


def _conserves(species, signs, coeffs):
    elems = set()
    for sp in species:
        for e in sp["elems"]:
            elems.add(e)
    for e in elems:
        if sum(signs[s] * coeffs[s] * sp["elems"].get(e, 0) for s, sp in enumerate(species)) != 0:
            return False
    return True


@dataclass
class EquationBalancingConfig(Config):
    n_species: int = 6
    n_elements: int = 5
    max_counter: int = 4
    min_counter: int = 2

    def apply_difficulty(self, level):
        self.n_species = int(self.n_species + 0.7 * level)
        self.max_counter = int(self.max_counter + 0.4 * level)


class EquationBalancing(Task):
    summary = "Find the smallest whole-number coefficients balancing a chemical reaction."
    config_cls = EquationBalancingConfig

    def generate_entry(self):
        for _ in range(600):
            n_species = max(4, min(8, self.config.n_species))
            max_counter = max(2, min(6, self.config.max_counter))
            n_react = random.randint(2, n_species - 2)
            signs = [1] * n_react + [-1] * (n_species - n_react)

            target = [random.randint(2, max_counter) for _ in range(n_species)]
            while max(target) - min(target) < 2:
                idx = random.randrange(n_species)
                target[idx] = random.randint(2, max_counter)
            g = 0
            for v in target:
                g = gcd(g, v)
            if g > 1:
                target = [v // g for v in target]

            n_pairs = n_species - 1
            pair_pool = [(r, p) for r in range(n_react) for p in range(n_react, n_species)]
            if len(pair_pool) < n_pairs:
                continue
            pairs = random.sample(pair_pool, n_pairs)
            elems = random.sample(list(ELEMENTS), n_pairs)
            species = [{"formula": "", "elems": {}} for _ in range(n_species)]
            for e, (r, p) in zip(elems, pairs):
                species[r]["elems"][e] = species[r]["elems"].get(e, 0) + target[p]
                species[p]["elems"][e] = species[p]["elems"].get(e, 0) + target[r]
            ok = True
            for sp in species:
                if not sp["elems"]:
                    ok = False
                    break
                counts = sp["elems"]
                ordered = sorted(counts.items(), key=lambda kv: ELEMENTS.index(kv[0]))
                sp["formula"] = "".join(f"{e}{n}" if n > 1 else e for e, n in ordered)
            if not ok:
                continue

            elems_present = sorted(set().union(*[set(sp["elems"]) for sp in species]), key=ELEMENTS.index)
            if len(elems_present) != n_species - 1:
                continue
            matrix = [[signs[s] * sp["elems"].get(e, 0) for s, sp in enumerate(species)] for e in elems_present]
            coeffs = _solve(matrix, n_species)
            if coeffs is None:
                continue
            if max(coeffs) > 9 or min(coeffs) < 1:
                continue
            if len(set(coeffs)) < 2:
                continue
            if list(coeffs) != list(target):
                continue
            if not _conserves(species, signs, coeffs):
                continue

            reactants = [sp["formula"] for sp in species[:n_react]]
            products = [sp["formula"] for sp in species[n_react:]]
            reaction = f"{' + '.join(reactants)} -> {' + '.join(products)}"
            ordered_formulas = [sp["formula"] for sp in species]
            return Entry(
                metadata=edict(
                    reaction=reaction,
                    species=ordered_formulas,
                    reactants=reactants,
                    products=products,
                    coeffs=coeffs,
                    elements=elems_present,
                ),
                answer=", ".join(str(c) for c in coeffs),
            )
        raise RuntimeError("EquationBalancing: could not build a balanced reaction")

    def render_prompt(self, m):
        order = ", ".join(m.species)
        return (
            f"Balance the following chemical reaction by finding the smallest positive whole-number "
            f"coefficients for every species. The species appear in this order: {order}.\n"
            f"{m.reaction}\n"
            f"Give the coefficients as a comma-separated list of integers, in the order the species "
            f"are written, with no spaces. Example format: 2,1,3,1"
        )

    def score_answer(self, answer, entry):
        text = str(answer).strip()
        if text == "":
            return 0.0
        try:
            parts = [int(p.strip()) for p in text.split(",")]
        except (ValueError, TypeError):
            return 0.0
        truth = entry.answer.split(",")
        if len(parts) != len(truth):
            return 0.0
        expected = [int(t.strip()) for t in truth]
        if parts == expected:
            return 1.0
        return 0.0
