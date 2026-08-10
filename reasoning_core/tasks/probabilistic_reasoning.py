import ast, json, random, re
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product

from gramforge import generate, init_grammar
from problog import get_evaluatable
from problog.program import PrologString
from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround
from reasoning_core.utils import score_space_ints


problog, eng = "problog", "eng"


def split(s):
    return s.split("|")


def peval(src):
    return get_evaluatable().create_from(PrologString(src)).evaluate()


def qprobs(src):
    return {str(k): float(v) for k, v in peval(src).items()}


def hidden_atoms(src):
    pat = r"(?m)^\s*(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)::\s*([a-z][a-z0-9_]*)\s*\."
    return re.findall(pat, src)


def sorted_lits(xs):
    return sorted(xs, key=lambda s: s.removeprefix("not "))


def mpe_solution(src):
    atoms = hidden_atoms(src)
    queries, keys = [], []

    for i, bits in enumerate(product([False, True], repeat=len(atoms))):
        name = f"mpe_{i}"
        body = ", ".join(a if b else rf"\+{a}" for a, b in zip(atoms, bits))
        lits = [a if b else f"not {a}" for a, b in zip(atoms, bits)]
        queries += [f"{name} :- {body}.", f"query({name})."]
        keys.append((name, sorted_lits(lits)))

    p = qprobs(src + "\n" + "\n".join(queries))
    ranked = sorted((p.get(k, 0.0), lits) for k, lits in keys)
    if len(ranked) > 1 and abs(ranked[-1][0] - ranked[-2][0]) < 1e-12:
        return None
    margin = ranked[-1][0] - ranked[-2][0] if len(ranked) > 1 else ranked[-1][0]
    return json.dumps(ranked[-1][1]), margin


def mpe_answer(src):
    sol = mpe_solution(src)
    return None if sol is None else sol[0]


def norm_lits(s):
    m = re.search(r"\[[^\]]*\]", s)
    if not m:
        return None
    try:
        return sorted_lits(map(str, ast.literal_eval(m.group(0))))
    except Exception:
        return None


def lit_options(src, shuffle_pairs=False):
    pairs = [[a, f"not {a}"] for a in hidden_atoms(src)]
    if shuffle_pairs:
        for pair in pairs:
            random.shuffle(pair)
    return [literal for pair in pairs for literal in pair]


def cmp_rules(cmp):
    return {
        "xx_vs_diff": ["a :- d1_x, d2_x.", "b :- d1_x, d2_y.", "b :- d1_y, d2_x."],
        "atleast_x_vs_yy": ["a :- d1_x.", "a :- d2_x.", "b :- d1_y, d2_y."],
        "same_vs_diff": ["a :- d1_x, d2_x.", "a :- d1_y, d2_y.", "b :- d1_x, d2_y.", "b :- d1_y, d2_x."],
        "xx_vs_yy": ["a :- d1_x, d2_x.", "b :- d1_y, d2_y."],
        "first_x_vs_first_y": ["a :- d1_x.", "b :- d1_y."],
    }[cmp]


def mpo_source(r, b, mode, cmp):
    n = r + b

    if mode == "wr":
        draw = [
            f"{r/n:.12g}::d1_x; {b/n:.12g}::d1_y.",
            f"{r/n:.12g}::d2_x; {b/n:.12g}::d2_y.",
        ]
    else:
        draw = [
            f"{r/n:.12g}::d1_x; {b/n:.12g}::d1_y.",
            f"{(r-1)/(n-1):.12g}::d2_x; {b/(n-1):.12g}::d2_y :- d1_x.",
            f"{r/(n-1):.12g}::d2_x; {(b-1)/(n-1):.12g}::d2_y :- d1_y.",
        ]

    return "\n".join(draw + cmp_rules(cmp) + ["query(a).", "query(b)."])


def mpo_label(r, b, mode, cmp):
    n = F(r + b)

    if mode == "wr":
        pxx = F(r, n) * F(r, n)
        pxy = F(r, n) * F(b, n)
        pyx = F(b, n) * F(r, n)
        pyy = F(b, n) * F(b, n)
    else:
        pxx = F(r, n) * F(r - 1, n - 1)
        pxy = F(r, n) * F(b, n - 1)
        pyx = F(b, n) * F(r, n - 1)
        pyy = F(b, n) * F(b - 1, n - 1)

    a, b = {
        "xx_vs_diff": (pxx, pxy + pyx),
        "atleast_x_vs_yy": (pxx + pxy + pyx, pyy),
        "same_vs_diff": (pxx + pyy, pxy + pyx),
        "xx_vs_yy": (pxx, pyy),
        "first_x_vs_first_y": (pxx + pxy, pyx + pyy),
    }[cmp]

    return "equal" if a == b else ("A" if a > b else "B")


def mpo_answer(src):
    p = qprobs(src)
    d = p["a"] - p["b"]
    return "equal" if abs(d) < 1e-12 else ("A" if d > 0 else "B")


def cmp_text(cmp, one, many, x, y):
    return {
        "xx_vs_diff": (f"both selected {many} are {x}", f"the selected {many} have different colors"),
        "atleast_x_vs_yy": (f"at least one selected {one} is {x}", f"both selected {many} are {y}"),
        "same_vs_diff": (f"the selected {many} have the same color", f"the selected {many} have different colors"),
        "xx_vs_yy": (f"both selected {many} are {x}", f"both selected {many} are {y}"),
        "first_x_vs_first_y": (f"the first selected {one} is {x}", f"the first selected {one} is {y}"),
    }[cmp]


def evidence_grammar():
    R = init_grammar([problog, eng], preprocess_template=lambda s: s)
    R("start(expr)", "{0}", "{0}")
    for atom in "abcdef":
        R("atom", atom, atom)
    R("pos(atom)", "{0}", "factor {0}", weight=3)
    R("pos(pos,pos)", "({0},{1})", "({0} and {1})", weight=1.2)
    R("pos(pos,pos)", "({0};{1})", "({0} or {1})")
    R("expr(pos)", "{0}", "{0}")
    R(
        "expr(pos,atom)",
        "({0},\\+{1})",
        "({0} holds and factor {1} is false)",
        weight=0.8,
    )
    return R


def evidence_instance(node, config=None):
    formula, text = node @ problog, node @ eng
    atoms = re.findall(r"\b[a-f]\b", formula)
    if len(atoms) != len(set(atoms)) or len(atoms) < 2:
        return None
    if config and not config.min_atoms <= len(atoms) <= config.max_atoms:
        return None
    probs = dict(zip(atoms, random.choices([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], k=len(atoms))))
    src = "\n".join(
        [f"{p}::{a}." for a, p in probs.items()]
        + [f"observed :- {formula}.", "evidence(observed,true)."]
    )
    english = "\n".join(
        [f"Factor {a} is independently true with probability {p}." for a, p in probs.items()]
        + [f"The observation holds exactly when {text}.", "We observe it.",
           "Which hidden fact values form the most probable complete explanation?"]
    )
    return src, english

def outcome_grammar(max_count=8, target=None):
    R = init_grammar([problog, eng], preprocess_template=lambda s: s)
    R("start(mpo)", "{0}", "{0}")

    for frame in [
        "A box|ball|balls|drawn",
        "A bag|token|tokens|sampled",
        "A jar|marble|marbles|picked",
        "A deck|card|cards|drawn",
        "A tray|tile|tiles|selected",
    ]:
        R("frame", frame, frame)

    for x, y in [
        ("red", "blue"),
        ("green", "yellow"),
        ("black", "white"),
        ("orange", "purple"),
        ("silver", "gold"),
    ]:
        R("palette", f"{x}|{y}", f"{x}|{y}")
        R("palette", f"{y}|{x}", f"{y}|{x}")

    for q in [
        "Which statement is more likely?",
    ]:
        R("ask", "", q)

    cmps = ["xx_vs_diff", "atleast_x_vs_yy", "same_vs_diff", "xx_vs_yy", "first_x_vs_first_y"]

    for r, b, mode, cmp in product(range(2, max_count + 1), range(2, max_count + 1), ["wr", "wor"], cmps):
        if target is None or mpo_label(r, b, mode, cmp) == target:
            R("design", f"{r}|{b}|{mode}|{cmp}", f"{r}|{b}|{mode}|{cmp}")

    def pl(frame, palette, design, ask):
        r, b, mode, cmp = split(design @ problog)
        return mpo_source(int(r), int(b), mode, cmp)

    def en(frame, palette, design, ask):
        box, one, many, verb = split(frame @ eng)
        x, y = split(palette @ eng)
        r, b, mode, cmp = split(design @ eng)
        mode_txt = (
            f"with the first {one} replaced before the second selection"
            if mode == "wr"
            else f"without replacing the first {one}"
        )
        A, B = cmp_text(cmp, one, many, x, y)
        return "\n".join([
            f"{box} contains {r} {x} {many} and {b} {y} {many}.",
            f"Two {many} are {verb} {mode_txt}.",
            ask @ eng,
            f"A: {A}.",
            f"B: {B}.",
        ])

    R("mpo(frame,palette,design,ask)", pl, en)
    return R

@dataclass
class MostProbableEvidenceConfig(Config):
    depth: int = 5
    min_atoms: int = 2
    max_atoms: int = 3
    max_attempts: int = 100
    min_margin: float = 0.03
    max_margin: float = 1.01

    def apply_difficulty(self, level):
        self.depth = sround(self.depth + level)
        self.max_atoms = sround(min(6, self.max_atoms + level))
        self.min_atoms = sround(min(4, self.min_atoms + level / 2))
        self.min_margin = max(0.005, self.min_margin * (0.75 ** level))
        self.max_margin = max(0.12, self.max_margin * (0.7 ** level))


class MostProbableEvidence(Task):
    summary = "Find the most probable configuration of hidden variables given evidence."
    def __init__(self, config=None):
        super().__init__(config=config or MostProbableEvidenceConfig())
        self.balancing_key_ratio = 1 / 3

    def generate_entry(self):
        for _ in range(self.config.max_attempts):
            node = generate(evidence_grammar(), depth=self.config.depth, min_depth=4)
            instance = evidence_instance(node, self.config)
            if instance is None:
                continue
            src, english = instance
            try:
                sol = mpe_solution(src)
            except Exception:
                continue
            if sol is None:
                continue
            answer, margin = sol
            if not self.config.min_margin <= margin <= self.config.max_margin:
                continue
            opts = lit_options(src, shuffle_pairs=True)
            lits = sorted_lits(map(str, json.loads(answer)))
            indices = [opts.index(x) for x in lits]
            pair_members = [index % 2 for index in indices]
            answer = " ".join(map(str, indices))
            return Entry(edict(problog=src, english=english, options=opts,
                               n_atoms=len(hidden_atoms(src)), margin=margin,
                               selected_pair_members=pair_members), answer)
        raise RuntimeError("Failed to generate probabilistic evidence task")

    def render_prompt(self, m):
        opts = "\n".join(f"{i}. {x}" for i, x in enumerate(m.options))
        return (
            f"{m.english}\n\nHidden fact values:\n{opts}\n\n"
            "Choose one value for each hidden factor. Answer with space-separated indexes."
        )

    def score_answer(self, answer, entry):
        return score_space_ints(answer, entry)

    def balancing_key(self, problem):
        positions = "".join(map(str, problem.metadata.selected_pair_members))
        return f"{problem.metadata.n_atoms}:{positions}"


@dataclass
class MostProbableOutcomeConfig(Config):
    max_count: int = 8
    depth: int = 5

    def apply_difficulty(self, level):
        self.max_count += level
        self.depth += level


class MostProbableOutcome(Task):
    summary = "Predict the most probable outcome or select hidden factor values in ProbLog."
    def __init__(self, config=None):
        super().__init__(config=config or MostProbableOutcomeConfig())
        self.balancing_key_ratio = 1 / 3

    def generate_entry(self):
        target = random.choice(["A", "B", "equal"])

        node = generate(outcome_grammar(self.config.max_count, target=target), depth=self.config.depth)
        src = node @ problog
        return Entry(
            metadata=edict(problog=src, english=node @ eng),
            answer=mpo_answer(src),
        )

    def render_prompt(self, m):
        return f"{m.english}\n\nThe answer is exactly one of: A, B, equal."

    def score_answer(self, answer, entry):
        m = re.fullmatch(r"\s*(A|B|equal)\.?\s*", answer, re.I)
        return float(bool(m) and m.group(1).lower() == entry.answer.lower())
