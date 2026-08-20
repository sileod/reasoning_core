# multistep_nli v1
from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
import random
from typing import Optional

from easydict import EasyDict as edict

from reasoning_core.template import Config, Entry, Task, render_payload, stochastic_rounding as sround
from reasoning_core.utils import parse_space_ints, score_space_ints


@dataclass(frozen=True)
class Atom:
    pred: str
    args: tuple
    sign: bool = True


@dataclass(frozen=True)
class Not:
    atom: Atom


@dataclass(frozen=True)
class PredSig:
    name: str
    arg_types: tuple
    verbalizers: tuple = ()


@dataclass(frozen=True)
class Rule:
    body: tuple
    head: Atom
    name: str = ""
    shape: str = ""


@dataclass(frozen=True)
class Denial:
    body: tuple


@dataclass
class Derivation:
    atom: Atom
    depth: int
    rule: Optional[object]
    parents: tuple


@dataclass
class Theory:
    facts: list
    rules: list
    denials: list
    pred_sigs: dict
    entities: dict
    domain_pack: str = "generic"


@dataclass
class ChaseResult:
    closure: set
    derivations: dict
    inconsistent: bool = False


@dataclass
class MultistepCase:
    theory: Theory
    res: ChaseResult
    label: str
    hyp: Atom
    target: Optional[Atom]
    derivation: Optional[Derivation]
    support_atoms: set
    lines: list
    source: dict
    surf: list
    live_rule_rate: float
    proof_rule_rate: float
    neutral_completion: Optional[Atom] = None

SUPPORTED_DOMAIN_PACKS = ("surface", "abstract", "spatial", "kinship")

@dataclass
class MultistepNLIConfig(Config):
    n_entities: int = 4
    n_unary_preds: int = 6
    n_binary_preds: int = 3
    n_facts: int = 4
    n_rules: int = 4
    max_depth: int = 3
    min_target_depth: int = 2
    max_target_depth: int = 3
    n_distractors: int = 1
    neutral_rate: float = 0.33
    contradiction_rate: float = 0.33
    group_entities: float = 0.5
    max_bin_size: int = 8
    domain_packs: tuple = SUPPORTED_DOMAIN_PACKS
    min_target_support_size: int = 2
    max_target_support_size: Optional[int] = 3
    
    def apply_difficulty(self, level):
        self.n_entities = min(8, self.n_entities + level)
        self.n_facts += level
        self.max_depth = sround(self.max_depth + 0.5 * level)
        self.n_rules += level
        self.n_distractors += 2 * level
        self.n_unary_preds += level
        self.n_binary_preds += level
        self.min_target_support_size = sround(min(5, 2 + 0.5 * level))
        if level > 0:
            self.max_target_support_size = None
        self.min_target_depth = sround(min(3, self.max_depth, 2 + 0.5 * level))
        self.max_target_depth = min(self.max_depth, self.min_target_depth + 1)

@dataclass
class MultistepAbductionConfig(MultistepNLIConfig):
    n_candidates: int = 6
    n_missing_facts: int = 1
    max_abduction_size: int = 1
    require_unique: bool = True

    def apply_difficulty(self, level):
        super().apply_difficulty(level)
        self.n_candidates += 2 * level
        self.max_abduction_size = sround(min(3, self.max_abduction_size + 0.5 * level))
        self.n_missing_facts = self.max_abduction_size

@dataclass
class LogicQAConfig(MultistepNLIConfig):
    answer_mode: str = "any"
    zero_answer_rate: float = 0.0


NAMES = ("alice", "bruno", "clara", "david", "elena", "farah", "george", "hannah")
OBJECTS = ("box", "key", "lamp", "map", "coin", "vase", "book", "ring")
PLACES = ("atrium", "lab", "office", "vault", "garden", "studio", "hall", "archive")

GEN_UNARY = (
    "approved", "careful", "trained", "trusted", "active", "verified", "alert",
    "eligible", "quiet", "skilled", "reliable", "flagged", "cleared", "busy",
)
GEN_BINARY = (
    "helps", "trusts", "advises", "checks", "visits", "guards", "reports_to",
    "contacts", "observes", "supports", "follows", "reviews",
)
SP_UNARY = ("marked", "fixed", "visible", "blocked", "safe", "fragile", "lit", "sealed")
ABS_UNARY = tuple(f"p{i}" for i in range(24))
ABS_BINARY = tuple(f"r{i}" for i in range(24))
TAG_WORDS = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "gamma", "kappa", "lambda", "omega", "sigma", "zeta",
)
REL_WORDS = (
    "alpha-linked", "beta-linked", "gamma-linked", "delta-related",
    "omega-connected", "sigma-associated", "kappa-linked", "zeta-related",
)



def is_var(x):
    return isinstance(x, str) and x.startswith("?")


def opposite(a):
    return Atom(a.pred, a.args, not a.sign)


def _subst(atom, env):
    return Atom(atom.pred, tuple(env.get(x, x) for x in atom.args), atom.sign)


def body_atoms(body):
    return [b for b in body if isinstance(b, Atom)]


def body_not_atoms(body):
    return [b.atom for b in body if isinstance(b, Not)]


def body_checks(body):
    return [b for b in body if not isinstance(b, (Atom, Not))]


def _match(pattern, fact, env):
    if pattern.pred != fact.pred or pattern.sign != fact.sign or len(pattern.args) != len(fact.args):
        return None
    env = dict(env)
    for p, v in zip(pattern.args, fact.args):
        if is_var(p):
            if p in env and env[p] != v:
                return None
            env[p] = v
        elif p != v:
            return None
    return env


def _rule_instances(rule, closure):
    index = defaultdict(list)
    for fact in closure:
        index[(fact.pred, fact.sign)].append(fact)

    atoms = [b for b in rule.body if isinstance(b, Atom)]
    checks = [b for b in rule.body if not isinstance(b, Atom)]

    def rec(i, env, parents):
        if i == len(atoms):
            for chk in checks:
                if chk[0] == "!=" and env.get(chk[1]) == env.get(chk[2]):
                    return
            head = _subst(rule.head, env)
            if any(is_var(a) for a in head.args):
                return
            yield head, parents
            return
        pat = atoms[i]
        for fact in index.get((pat.pred, pat.sign), ()):
            env2 = _match(pat, fact, env)
            if env2 is not None:
                yield from rec(i + 1, env2, parents + (fact,))

    yield from rec(0, {}, ())


def _vars_in_atom(atom):
    return {a for a in atom.args if is_var(a)}


def _check_naf_safety(rule):
    bound = set().union(*(_vars_in_atom(a) for a in body_atoms(rule.body))) if body_atoms(rule.body) else set()
    needed = set(_vars_in_atom(rule.head))
    if body_not_atoms(rule.body):
        needed |= set().union(*(_vars_in_atom(a) for a in body_not_atoms(rule.body)))
    for chk in body_checks(rule.body):
        if chk[0] == "!=":
            needed |= {x for x in chk[1:] if is_var(x)}
    if needed - bound:
        raise ValueError("unsafe NAF rule")


def _rule_instances_naf(rule, closure):
    _check_naf_safety(rule)
    index = defaultdict(list)
    for fact in closure:
        index[(fact.pred, fact.sign)].append(fact)

    atoms = body_atoms(rule.body)
    not_atoms = body_not_atoms(rule.body)
    checks = body_checks(rule.body)

    def rec(i, env, parents):
        if i == len(atoms):
            for chk in checks:
                if chk[0] == "!=" and env.get(chk[1]) == env.get(chk[2]):
                    return
            for atom in not_atoms:
                inst = _subst(atom, env)
                if any(is_var(a) for a in inst.args) or inst in closure:
                    return
            head = _subst(rule.head, env)
            if any(is_var(a) for a in head.args):
                return
            yield head, parents
            return
        pat = atoms[i]
        for fact in index.get((pat.pred, pat.sign), ()):
            env2 = _match(pat, fact, env)
            if env2 is not None:
                yield from rec(i + 1, env2, parents + (fact,))

    yield from rec(0, {}, ())


def pred_key(atom):
    return atom.pred, len(atom.args), atom.sign


def _rule_predicates(rule):
    preds = {pred_key(rule.head)}
    preds.update(pred_key(a) for a in body_atoms(rule.body))
    preds.update(pred_key(a) for a in body_not_atoms(rule.body))
    return preds


def _has_negative_scc(preds, edges):
    graph = defaultdict(list)
    for src, dst, negative in edges:
        graph[src].append((dst, negative))

    index = 0
    stack, on_stack, indices, low, comps = [], set(), {}, {}, []

    def strong(v):
        nonlocal index
        indices[v] = low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w, _ in graph.get(v, ()):
            if w not in indices:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp = set()
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.add(w)
                if w == v:
                    break
            comps.append(comp)

    for p in preds:
        if p not in indices:
            strong(p)
    return any(negative and src in comp and dst in comp for comp in comps for src, dst, negative in edges)


def stratify_rules(rules):
    preds, edges = set(), []
    for rule in rules:
        _check_naf_safety(rule)
        head = pred_key(rule.head)
        preds.update(_rule_predicates(rule))
        for atom in body_atoms(rule.body):
            edges.append((pred_key(atom), head, False))
        for atom in body_not_atoms(rule.body):
            edges.append((pred_key(atom), head, True))
    strata = {p: 0 for p in preds}
    if _has_negative_scc(preds, edges):
        return None
    for _ in range(max(1, len(preds))):
        changed = False
        for src, dst, negative in edges:
            need = strata[src] + (1 if negative else 0)
            if strata[dst] < need:
                strata[dst] = need
                changed = True
        if not changed:
            return strata
    for src, dst, negative in edges:
        if strata[dst] < strata[src] + (1 if negative else 0):
            return None
    return strata


def chase(theory, max_depth=None):
    closure = set(theory.facts)
    deriv = {a: Derivation(a, 0, None, ()) for a in closure}

    def denial_hit():
        for p in list(closure):
            if opposite(p) in closure:
                return True
        for d in theory.denials:
            dummy = Rule(d.body, Atom("__false__", ()))
            if next(_rule_instances(dummy, closure), None):
                return True
        return False

    if denial_hit():
        return ChaseResult(closure, deriv, True)

    changed = True
    while changed:
        changed = False
        for rule in theory.rules:
            for head, parents in _rule_instances(rule, closure):
                depth = 1 + max(deriv[p].depth for p in parents) if parents else 1
                if max_depth is not None and depth > max_depth:
                    continue
                old = deriv.get(head)
                if old is None or depth < old.depth:
                    closure.add(head)
                    deriv[head] = Derivation(head, depth, rule, parents)
                    changed = True
        if denial_hit():
            return ChaseResult(closure, deriv, True)
    return ChaseResult(closure, deriv, False)


def naf_chase(theory, max_depth=None):
    if max_depth is not None:
        raise ValueError("naf_chase requires max_depth=None")
    strata = stratify_rules(theory.rules)
    if strata is None:
        raise ValueError("unstratified NAF program")
    closure = set(theory.facts)
    deriv = {a: Derivation(a, 0, None, ()) for a in closure}

    def denial_hit():
        for p in list(closure):
            if opposite(p) in closure:
                return True
        for d in theory.denials:
            dummy = Rule(d.body, Atom("__false__", ()))
            if next(_rule_instances_naf(dummy, closure), None):
                return True
        return False

    if denial_hit():
        return ChaseResult(closure, deriv, True)

    by_stratum = defaultdict(list)
    for rule in theory.rules:
        by_stratum[strata.get(pred_key(rule.head), 0)].append(rule)

    for s in range(max(by_stratum.keys(), default=-1) + 1):
        changed = True
        while changed:
            changed = False
            for rule in by_stratum.get(s, ()):
                for head, parents in _rule_instances_naf(rule, closure):
                    depth = 1 + max(deriv[p].depth for p in parents) if parents else 1
                    if max_depth is not None and depth > max_depth:
                        continue
                    old = deriv.get(head)
                    if old is None or depth < old.depth:
                        closure.add(head)
                        deriv[head] = Derivation(head, depth, rule, parents)
                        changed = True
            if denial_hit():
                return ChaseResult(closure, deriv, True)
    return ChaseResult(closure, deriv, False)


def _sigs(names, arity, typ="person"):
    return {p: PredSig(p, (typ,) * arity, (p.replace("_", " "),)) for p in names}


def _signed_converse(p, q):
    out = []
    for sign in (True, False):
        out.append(Rule((Atom(p, ("?x", "?y"), sign),), Atom(q, ("?y", "?x"), sign), "spatial", "converse"))
        out.append(Rule((Atom(q, ("?x", "?y"), sign),), Atom(p, ("?y", "?x"), sign), "spatial", "converse"))
    return out


def _rule_key(rule):
    return frozenset(rule.body), rule.head


def _domain_pack(name, cfg):
    n = min(max(3, cfg.n_entities), 7)
    if name == "abstract":
        ents = {"entity": tuple(x.title() for x in NAMES[:n])}
        unary = ABS_UNARY[: cfg.n_unary_preds]
        binary = ABS_BINARY[: cfg.n_binary_preds]
        return ents, {**_sigs(unary, 1, "entity"), **_sigs(binary, 2, "entity")}, [], []
    if name == "surface":
        ents = {"person": NAMES[:n]}
        unary = GEN_UNARY[: cfg.n_unary_preds]
        binary = GEN_BINARY[: cfg.n_binary_preds]
        return ents, {**_sigs(unary, 1), **_sigs(binary, 2)}, [], []
    if name == "spatial":
        ents = {"item": OBJECTS[:n], "place": PLACES[: max(3, n // 2)]}
        unary = SP_UNARY[: cfg.n_unary_preds]
        binary = ("left_of", "right_of", "above", "below", "inside", "contains", "disjoint")
        sigs = {**_sigs(unary, 1, "item"), **_sigs(binary[: cfg.n_binary_preds + 4], 2, "item")}
        bg = []
        bg += _signed_converse("left_of", "right_of")
        bg += _signed_converse("above", "below")
        bg += _signed_converse("inside", "contains")
        bg += [
            Rule((Atom("inside", ("?x", "?y")), Atom("inside", ("?y", "?z"))), Atom("inside", ("?x", "?z")), "spatial", "composition"),
            Rule((Atom("left_of", ("?x", "?y")), Atom("left_of", ("?y", "?z"))), Atom("left_of", ("?x", "?z")), "spatial", "composition"),
            Rule((Atom("above", ("?x", "?y")), Atom("above", ("?y", "?z"))), Atom("above", ("?x", "?z")), "spatial", "composition"),
        ]
        for sign in (True, False):
            bg.append(Rule((Atom("disjoint", ("?x", "?y"), sign),), Atom("disjoint", ("?y", "?x"), sign), "spatial", "converse"))
        irreflexive = ("left_of", "right_of", "above", "below", "inside", "contains", "disjoint")
        asymmetric = ("left_of", "right_of", "above", "below", "inside", "contains")
        denials = [Denial((Atom(p, ("?x", "?x")),)) for p in irreflexive]
        denials += [Denial((Atom(p, ("?x", "?y")), Atom(p, ("?y", "?x")))) for p in asymmetric]
        denials += [
            Denial((Atom("inside", ("?x", "?y")), Atom("disjoint", ("?x", "?y")))),
            Denial((Atom("contains", ("?x", "?y")), Atom("disjoint", ("?x", "?y")))),
        ]
        return ents, sigs, bg, denials
    ents = {"person": NAMES[:n]}
    if name == "kinship":
        unary = ("adult", "minor", "kind", "patient", "careful", "trusted", "female", "male")
        binary = ("parent", "ancestor", "sibling", "spouse", "aunt_or_uncle", "helps", "trusts")
        sigs = {**_sigs(unary[: cfg.n_unary_preds], 1), **_sigs(binary[: cfg.n_binary_preds + 5], 2)}
        bg = [
            Rule((Atom("parent", ("?x", "?y")),), Atom("ancestor", ("?x", "?y")), "kin", "bridge"),
            Rule((Atom("parent", ("?x", "?y")), Atom("ancestor", ("?y", "?z"))), Atom("ancestor", ("?x", "?z")), "kin", "composition"),
            Rule((Atom("parent", ("?p", "?x")), Atom("parent", ("?p", "?y")), ("!=", "?x", "?y")), Atom("sibling", ("?x", "?y")), "kin", "sibling"),
            Rule((Atom("sibling", ("?x", "?y")),), Atom("sibling", ("?y", "?x")), "kin", "converse"),
            Rule((Atom("spouse", ("?x", "?y")),), Atom("spouse", ("?y", "?x")), "kin", "converse"),
            Rule((Atom("parent", ("?x", "?y")), Atom("sibling", ("?x", "?z"))), Atom("aunt_or_uncle", ("?z", "?y")), "kin", "bridge"),
        ]
        denials = [
            Denial((Atom("male", ("?x",)), Atom("female", ("?x",)))),
            Denial((Atom("adult", ("?x",)), Atom("minor", ("?x",)))),
        ]
        irreflexive = ("parent", "ancestor", "sibling", "spouse", "aunt_or_uncle")
        asymmetric = ("parent", "ancestor")
        denials += [Denial((Atom(p, ("?x", "?x")),)) for p in irreflexive]
        denials += [Denial((Atom(p, ("?x", "?y")), Atom(p, ("?y", "?x")))) for p in asymmetric]
        kin_conflicts = [
            ("parent", "sibling"),
            ("parent", "spouse"),
            ("ancestor", "sibling"),
            ("ancestor", "spouse"),
            ("sibling", "spouse"),
        ]
        denials += [
            Denial((Atom(a, ("?x", "?y")), Atom(b, ("?x", "?y"))))
            for a, b in kin_conflicts
        ]
        return ents, sigs, bg, denials
    unary = GEN_UNARY[: cfg.n_unary_preds]
    binary = GEN_BINARY[: cfg.n_binary_preds]
    return ents, {**_sigs(unary, 1), **_sigs(binary, 2)}, [], []


def _vars_for(sig):
    return tuple(f"?{chr(120 + i)}" for i in range(len(sig.arg_types)))


def _random_atom(sigs, entities, arity=None, ground=False, sign=None, pred=None):
    choices = [s for s in sigs.values() if arity is None or len(s.arg_types) == arity]
    sig = sigs[pred] if pred else random.choice(choices)
    args = []
    for i, typ in enumerate(sig.arg_types):
        args.append(random.choice(entities[typ]) if ground else f"?{chr(120 + i)}")
    return Atom(sig.name, tuple(args), random.choice([True, False]) if sign is None else sign)


def _fresh_fact(theory, used):
    for _ in range(100):
        a = _random_atom(theory.pred_sigs, theory.entities, ground=True, sign=random.random() > 0.18)
        if len(a.args) == 2 and a.args[0] == a.args[1]:
            continue
        if a not in used and opposite(a) not in used:
            used.add(a)
            return a
    return None


def _sample_rule(sigs):
    unaries = [s for s in sigs.values() if len(s.arg_types) == 1]
    binaries = [s for s in sigs.values() if len(s.arg_types) == 2]
    shapes = []
    if len(unaries) >= 2:
        shapes.append("u_imp")
    if len(unaries) >= 3:
        shapes += ["u_and", "signed"]
    if binaries and len(unaries) >= 2:
        shapes += ["rel_y", "rel_x"]
    if len(binaries) >= 2:
        shapes.append("converse")
    if len(binaries) >= 3:
        shapes.append("composition")
    if not shapes:
        raise ValueError("not enough predicates to sample a rule")
    shape = random.choice(shapes)
    neg_head = shape == "signed" or random.random() < 0.15
    if shape == "u_imp":
        if len(unaries) < 2:
            raise ValueError("need at least two unary predicates")
        a, b = random.sample(unaries, 2)
        return Rule((Atom(a.name, ("?x",)),), Atom(b.name, ("?x",), not neg_head), shape=shape)
    if shape in ("u_and", "signed"):
        if len(unaries) < 3:
            raise ValueError("need at least three unary predicates")
        a, b, c = random.sample(unaries, 3)
        return Rule((Atom(a.name, ("?x",)), Atom(b.name, ("?x",))), Atom(c.name, ("?x",), not neg_head), shape=shape)
    if shape == "rel_y":
        r = random.choice(binaries); a, b = random.sample(unaries, 2)
        return Rule((Atom(r.name, ("?x", "?y")), Atom(a.name, ("?y",))), Atom(b.name, ("?x",), not neg_head), shape=shape)
    if shape == "rel_x":
        r = random.choice(binaries); a, b = random.sample(unaries, 2)
        return Rule((Atom(r.name, ("?x", "?y")), Atom(a.name, ("?x",))), Atom(b.name, ("?y",), not neg_head), shape=shape)
    if shape == "converse":
        r, s = random.sample(binaries, 2)
        return Rule((Atom(r.name, ("?x", "?y")),), Atom(s.name, ("?y", "?x"), not neg_head), shape=shape)
    if shape == "composition":
        r, s, t = random.sample(binaries, 3)
        return Rule((Atom(r.name, ("?x", "?y")), Atom(s.name, ("?y", "?z"))), Atom(t.name, ("?x", "?z"), not neg_head), shape=shape)
    raise AssertionError(f"unknown rule shape: {shape}")


def _bad_rule(rule):
    if any(isinstance(b, Atom) and b == rule.head for b in rule.body):
        return True
    conflicts = {frozenset(("male", "female")), frozenset(("adult", "minor"))}
    positive = defaultdict(set)
    for b in rule.body:
        if isinstance(b, Atom) and b.sign:
            positive[b.args].add(b.pred)
    return any(pair <= preds for preds in positive.values() for pair in conflicts)


def _plant_backbone(theory, sigs, entities, cfg=None):
    unaries = [s for s in sigs.values() if len(s.arg_types) == 1]
    binaries = [s for s in sigs.values() if len(s.arg_types) == 2]
    support = int(getattr(cfg, "min_target_support_size", 2))
    if support >= 3 and len(unaries) >= 2 * support - 1:
        chosen = random.sample(unaries, 2 * support - 1)
        inputs, outputs = chosen[:support], chosen[support:]
        x = random.choice(entities[inputs[0].arg_types[0]])
        theory.facts += [Atom(a.name, (x,)) for a in inputs]
        previous = inputs[0]
        for other, out in zip(inputs[1:], outputs):
            theory.rules.append(Rule(
                (Atom(previous.name, ("?x",)), Atom(other.name, ("?x",))),
                Atom(out.name, ("?x",)), "backbone", "u_and",
            ))
            previous = out
        return {
            "kind": "wide_conjunctive", "entities": (x,),
            "unaries": tuple(s.name for s in chosen), "binaries": (),
        }
    options = ["unary"]
    if len(unaries) >= 2 and binaries:
        options.append("bridge")
    if unaries and len(binaries) >= 3:
        options.append("composition")
    if len(unaries) >= 4:
        options.append("conjunctive")
    kind = random.choice(options)

    if kind == "bridge":
        r = random.choice(binaries)
        a, b, c = random.sample(unaries, 3 if len(unaries) >= 3 else 2) if len(unaries) >= 3 else (*random.sample(unaries, 2), None)
        x, y = [random.choice(entities[t]) for t in r.arg_types]
        if x == y:
            y = random.choice([e for e in entities[r.arg_types[1]] if e != x] or entities[r.arg_types[1]])
        theory.facts += [Atom(r.name, (x, y)), Atom(a.name, (y,))]
        theory.rules.append(Rule((Atom(r.name, ("?x", "?y")), Atom(a.name, ("?y",))), Atom(b.name, ("?x",)), "backbone", "rel_y"))
        if c:
            theory.rules.append(Rule((Atom(b.name, ("?x",)),), Atom(c.name, ("?x",)), "backbone", "u_imp"))
        return {"kind": kind, "entities": (x, y), "unaries": tuple(s.name for s in (a, b) + ((c,) if c else ())), "binaries": (r.name,)}

    if kind == "composition":
        r, s, t = random.sample(binaries, 3)
        a = random.choice(unaries)
        x = random.choice(entities[r.arg_types[0]])
        y = random.choice(entities[r.arg_types[1]])
        z = random.choice(entities[s.arg_types[1]])
        if x == y:
            y = random.choice([e for e in entities[r.arg_types[1]] if e != x] or entities[r.arg_types[1]])
        if y == z:
            z = random.choice([e for e in entities[s.arg_types[1]] if e != y] or entities[s.arg_types[1]])
        theory.facts += [Atom(r.name, (x, y)), Atom(s.name, (y, z))]
        theory.rules += [
            Rule((Atom(r.name, ("?x", "?y")), Atom(s.name, ("?y", "?z"))), Atom(t.name, ("?x", "?z")), "backbone", "composition"),
            Rule((Atom(t.name, ("?x", "?z")),), Atom(a.name, ("?x",)), "backbone", "rel_to_unary"),
        ]
        return {"kind": kind, "entities": (x, y, z), "unaries": (a.name,), "binaries": (r.name, s.name, t.name)}

    if kind == "conjunctive":
        a, b, c, d = random.sample(unaries, 4)
        x = random.choice(entities[a.arg_types[0]])
        theory.facts += [Atom(a.name, (x,)), Atom(b.name, (x,))]
        theory.rules += [
            Rule((Atom(a.name, ("?x",)), Atom(b.name, ("?x",))), Atom(c.name, ("?x",)), "backbone", "u_and"),
            Rule((Atom(c.name, ("?x",)),), Atom(d.name, ("?x",)), "backbone", "u_imp"),
        ]
        return {"kind": kind, "entities": (x,), "unaries": tuple(s.name for s in (a, b, c, d)), "binaries": ()}

    if len(unaries) >= 4:
        chain = random.sample(unaries, random.randint(3, min(5, len(unaries))))
        ent = random.choice(entities[chain[0].arg_types[0]])
        theory.facts.append(Atom(chain[0].name, (ent,)))
        for i, (a, b) in enumerate(zip(chain, chain[1:])):
            sign = True if i + 2 < len(chain) else random.random() > 0.25
            theory.rules.append(Rule((Atom(a.name, ("?x",)),), Atom(b.name, ("?x",), sign), "backbone", "u_imp"))
        return {"kind": "unary", "entities": (ent,), "unaries": tuple(s.name for s in chain), "binaries": ()}
    return None


def _add_near_misses(theory, sigs, entities, backbone, n=4):
    if not backbone:
        return
    unaries = [s.name for s in sigs.values() if len(s.arg_types) == 1]
    binaries = [s.name for s in sigs.values() if len(s.arg_types) == 2]
    es = list(next(iter(entities.values())))
    ents = tuple(backbone.get("entities", ()))
    us = tuple(backbone.get("unaries", ()))
    bs = tuple(backbone.get("binaries", ()))
    seen = set(theory.facts)
    seen_rules = {_rule_key(r) for r in theory.rules}
    def add_fact(a):
        if a in seen or opposite(a) in seen:
            return False
        theory.facts.append(a)
        seen.add(a)
        return True
    added = 0
    for _ in range(max(100, 20 * n)):
        if added >= n:
            break
        kind = random.choice(["wrong_entity", "wrong_pred", "polarity", "near_rule"])
        if kind == "wrong_entity" and bs and us and len(ents) >= 2:
            x, y = ents[0], ents[1]
            other = random.choice([e for e in es if e not in {x, y}] or es)
            ok = add_fact(Atom(bs[0], (x, other)))
            ok = add_fact(Atom(us[0], (other,), random.random() > 0.35)) or ok
        elif kind == "wrong_pred" and len(bs) >= 1 and len(binaries) >= 2 and len(ents) >= 2:
            alt = random.choice([b for b in binaries if b not in bs] or binaries)
            ok = add_fact(Atom(alt, (ents[0], ents[1])))
        elif kind == "polarity" and us and ents:
            ok = add_fact(Atom(us[0], (random.choice(ents),), False))
        elif kind == "near_rule" and us:
            a = random.choice(us)
            b = random.choice([u for u in unaries if u not in us] or unaries)
            c = random.choice([u for u in unaries if u != b] or unaries)
            rule = Rule((Atom(a, ("?x",)), Atom(b, ("?x",))), Atom(c, ("?x",)), "near_miss", "u_and")
            k = _rule_key(rule)
            ok = k not in seen_rules and not _bad_rule(rule)
            if ok:
                theory.rules.append(rule)
                seen_rules.add(k)
        else:
            continue
        added += int(ok)


def sample_theory(cfg):
    pack = random.choice(tuple(cfg.domain_packs))
    entities, sigs, bg, denials = _domain_pack(pack, cfg)
    theory = Theory([], [], denials, sigs, entities, pack)
    used = set()
    backbone = None
    if pack in {"abstract", "surface"}:
        backbone = _plant_backbone(theory, sigs, entities, cfg)
        _add_near_misses(theory, sigs, entities, backbone, cfg.n_distractors)
    elif pack == "spatial":
        es = entities["item"]
        chain = random.sample(es, min(len(es), max(3, int(cfg.min_target_support_size) + 1)))
        rel = random.choice(["left_of", "above", "inside"])
        theory.facts += [Atom(rel, pair) for pair in zip(chain, chain[1:])]
    elif pack == "kinship":
        es = entities["person"]
        chain = random.sample(es, min(len(es), max(3, int(cfg.min_target_support_size) + 1)))
        theory.facts += [Atom("parent", pair) for pair in zip(chain, chain[1:])]
    used.update(theory.facts)
    for _ in range(cfg.n_facts):
        if f := _fresh_fact(theory, used):
            theory.facts.append(f)
    theory.rules = list(bg) + theory.rules
    if pack in {"abstract", "surface"}:
        used_rules = {_rule_key(r) for r in theory.rules}
        for _ in range(cfg.n_rules):
            for _ in range(20):
                rule = _sample_rule(sigs)
                k = _rule_key(rule)
                if k not in used_rules and not _bad_rule(rule):
                    theory.rules.append(rule)
                    used_rules.add(k)
                    break
    return theory


@dataclass
class DefeasibleNLIConfig(MultistepNLIConfig):
    domain_packs: tuple = ("surface", "abstract")
    naf_rule_rate: float = 0.5
    exception_rate: float = 0.35
    min_naf_rules_in_proof: int = 1

    def apply_difficulty(self, level):
        self.n_entities = min(7, self.n_entities + level)
        self.n_facts += level
        self.n_rules += level
        self.n_distractors += 2 * level
        self.max_depth = 3
        self.min_target_depth = 2
        self.max_target_depth = 3
        if level >= 2:
            self.naf_rule_rate = min(0.85, self.naf_rule_rate + 0.15)
            self.exception_rate = min(0.50, self.exception_rate + 0.05)


def _naf_roles(pack):
    if pack == "abstract":
        return {
            "trained": "p0", "flagged": "p1", "trusted": "p2", "blocked": "p3",
            "approved": "p4", "bird": "p5", "penguin": "p6", "ab_bird": "p7",
            "careful": "p8", "helps": "r0",
        }
    return {
        "trained": "trained", "flagged": "flagged", "trusted": "trusted",
        "blocked": "blocked", "approved": "approved", "bird": "bird",
        "penguin": "penguin", "ab_bird": "ab_bird", "careful": "careful",
        "helps": "helps",
    }


def sample_naf_theory(cfg):
    pack = random.choice(tuple(cfg.domain_packs))
    typ = "entity" if pack == "abstract" else "person"
    n = min(max(3, cfg.n_entities), 7)
    ents = {typ: tuple(x.title() for x in NAMES[:n])} if pack == "abstract" else {typ: NAMES[:n]}
    roles = _naf_roles(pack)
    sigs = {
        roles[k]: PredSig(roles[k], (typ,), ())
        for k in ("trained", "flagged", "trusted", "blocked", "approved", "bird", "penguin", "ab_bird", "careful")
    }
    sigs[roles["helps"]] = PredSig(roles["helps"], (typ, typ), ())
    people = list(ents[typ])
    x, y, z = random.sample(people, 3)
    trust_exc = roles["flagged"] if random.random() < 0.7 else roles["blocked"]
    approve_exc = roles["blocked"] if trust_exc == roles["flagged"] else roles["flagged"]
    rules = [
        Rule((Atom(roles["trained"], ("?x",)), Not(Atom(trust_exc, ("?x",)))), Atom(roles["trusted"], ("?x",)), "naf", "default_trusted"),
        Rule((Atom(roles["trusted"], ("?x",)), Not(Atom(approve_exc, ("?x",)))), Atom(roles["approved"], ("?x",)), "naf", "default_approved"),
        Rule((Atom(trust_exc, ("?x",)),), Atom(roles["trusted"], ("?x",), False), "naf", "exception_negative"),
        Rule((Atom(roles["trusted"], ("?x",), False), Not(Atom(approve_exc, ("?x",)))), Atom(roles["approved"], ("?x",), False), "naf", "default_reject"),
    ]
    optional = [
        Rule((Atom(roles["penguin"], ("?x",)),), Atom(roles["ab_bird"], ("?x",)), "naf", "abnormality"),
        Rule((Atom(roles["bird"], ("?x",)), Not(Atom(roles["ab_bird"], ("?x",)))), Atom(roles["approved"], ("?x",)), "naf", "bird_default"),
        Rule((Atom(roles["helps"], ("?x", "?y")), Atom(roles["careful"], ("?y",)), Not(Atom(trust_exc, ("?y",)))), Atom(roles["trusted"], ("?x",)), "naf", "bridge_default"),
        Rule((Atom(roles["trained"], ("?x",)), Not(Atom(approve_exc, ("?x",)))), Atom(roles["careful"], ("?x",)), "naf", "careful_default"),
        Rule((Atom(roles["helps"], ("?x", "?y")), Atom(roles["trusted"], ("?y",)), Not(Atom(approve_exc, ("?x",)))), Atom(roles["approved"], ("?x",)), "naf", "helped_by_trusted"),
    ]
    random.shuffle(optional)
    for rule in optional:
        if len(rules) >= cfg.n_rules + 3:
            break
        if random.random() < cfg.naf_rule_rate or len(rules) < 6:
            rules.append(rule)
    facts = [
        Atom(roles["trained"], (x,)),
        Atom(roles["trained"], (y,)),
        Atom(trust_exc, (y,)),
        Atom(roles["bird"], (x,)),
        Atom(roles["bird"], (y,)),
        Atom(roles["penguin"], (y,)),
        Atom(roles["helps"], (x, y)),
        Atom(roles["careful"], (y,)),
    ]
    if random.random() < cfg.exception_rate:
        facts.append(Atom(approve_exc, (z,)))
    if random.random() < cfg.exception_rate:
        facts.append(Atom(roles["penguin"], (z,)))
    used = set(facts)
    for _ in range(max(0, cfg.n_facts - 2)):
        fact = _fresh_fact(Theory(facts, rules, [], sigs, ents, pack), used)
        if fact:
            facts.append(fact)
    theory = Theory(facts, rules[: max(2, min(len(rules), cfg.n_rules + 3))], [], sigs, ents, pack)
    if stratify_rules(theory.rules) is None:
        raise RuntimeError("sampled unstratified NAF theory")
    return theory


def support_atoms(atom, deriv):
    d = deriv[atom]
    if d.rule is None:
        return {atom}
    out = set()
    for p in d.parents:
        out |= support_atoms(p, deriv)
    return out


def support_sources(atom, deriv, source):
    d = deriv[atom]
    if d.rule is None:
        return {source[atom]} if atom in source else set()
    out = {source[d.rule]} if d.rule in source else set()
    for p in d.parents:
        out |= support_sources(p, deriv, source)
    return out


def _focused_theory(theory, res, targets, n_distractors):
    """Keep complete proof cones plus a bounded sample of irrelevant statements."""
    facts, rules = set(), set()

    def visit(atom):
        d = res.derivations[atom]
        if d.rule is None:
            facts.add(atom)
            return
        rules.add(d.rule)
        for parent in d.parents:
            visit(parent)

    for target in targets:
        visit(target)
    extras = [
        (kind, item)
        for kind, items, used in (
            ("fact", theory.facts, facts), ("rule", theory.rules, rules)
        )
        for item in items if item not in used
    ]
    random.shuffle(extras)
    for kind, item in extras[:max(0, int(n_distractors))]:
        (facts if kind == "fact" else rules).add(item)
    return Theory(
        [x for x in theory.facts if x in facts],
        [x for x in theory.rules if x in rules],
        theory.denials, theory.pred_sigs, theory.entities, theory.domain_pack,
    )


def hard_target(a, deriv, cfg):
    d = deriv[a].depth
    s = len(support_atoms(a, deriv))
    return (
        cfg.min_target_depth <= d <= min(cfg.max_target_depth, cfg.max_depth)
        and s >= cfg.min_target_support_size
        and (cfg.max_target_support_size is None or s <= cfg.max_target_support_size)
    )


def _all_ground_atoms(theory, signs=(True, False)):
    for sig in theory.pred_sigs.values():
        pools = [theory.entities[t] for t in sig.arg_types]
        for args in itertools.product(*pools):
            if len(args) == 2 and args[0] == args[1]:
                continue
            for sign in signs:
                yield Atom(sig.name, tuple(args), sign)


def close_with(theory, extra):
    t = Theory(list(theory.facts) + list(extra), theory.rules, theory.denials, theory.pred_sigs, theory.entities, theory.domain_pack)
    return chase(t, max_depth=None)


def choose_example(theory, res, cfg, label_signs=None, label_depths=None):
    label_signs = label_signs or Counter()
    label_depths = label_depths or Counter()
    target_depths = tuple(range(cfg.min_target_depth, min(cfg.max_target_depth, cfg.max_depth) + 1))
    derived = [a for a in res.closure if res.derivations[a].depth > 0]
    non_direct = [
        a for a in derived
        if a not in theory.facts
        and opposite(a) not in theory.facts
        and hard_target(a, res.derivations, cfg)
    ]

    def prefer_support_range(pool, target=lambda x: x):
        return [h for h in pool if hard_target(target(h), res.derivations, cfg)]

    r = random.random()
    wants = "neutral" if r < cfg.neutral_rate else "contradiction" if r < cfg.neutral_rate + cfg.contradiction_rate else "entailment"
    labels = [wants, "entailment", "contradiction", "neutral"]
    seen = set()
    def balanced(label, pool):
        want = True if label_signs[(label, True)] <= label_signs[(label, False)] else False
        same = [h for h in pool if h.sign == want]
        return same or pool
    def choose_by_depth(label, pool, target=lambda x: x):
        by_depth = defaultdict(list)
        for h in pool:
            by_depth[res.derivations[target(h)].depth].append(h)
        available = [d for d in target_depths if by_depth[d]]
        if not available:
            return None
        min_fill = min(label_depths[(label, d)] for d in available)
        least = [d for d in available if label_depths[(label, d)] == min_fill]
        d = random.choices(least, weights=[d * d for d in least])[0]
        return random.choice(by_depth[d])

    for label in [x for x in labels if not (x in seen or seen.add(x))]:
        if label == "entailment":
            pool = prefer_support_range(balanced(label, non_direct))
            if pool:
                h = choose_by_depth(label, pool)
                if h is None:
                    continue
                return label, h, res.derivations[h], support_atoms(h, res.derivations)
        if label == "contradiction":
            pool = prefer_support_range(
                balanced(label, [opposite(a) for a in non_direct if opposite(a) not in theory.facts]),
                opposite,
            )
            if pool:
                h = choose_by_depth(label, pool, opposite)
                if h is None:
                    continue
                d = res.derivations[opposite(h)]
                return label, h, d, support_atoms(opposite(h), res.derivations)
        if label == "neutral":
            closure = res.closure
            near_preds = Counter(a.pred for a in derived or closure)
            pool = sorted(
                [a for a in _all_ground_atoms(theory) if a not in closure and opposite(a) not in closure],
                key=lambda a: -near_preds[a.pred],
            )
            pool = balanced(label, pool)
            near, rest = pool[: min(30, len(pool))], pool[min(30, len(pool)) :]
            random.shuffle(near)
            pool = near + rest
            for h in pool[:80]:
                if not close_with(theory, [h]).inconsistent and not close_with(theory, [opposite(h)]).inconsistent:
                    return label, h, None, set()
    return None


def _label_for_atom(res, hyp):
    if hyp in res.closure:
        return "entailment"
    if opposite(hyp) in res.closure:
        return "contradiction"
    return "neutral"


def _naf_rules_in_derivation(atom, deriv):
    return sum(1 for r in derivation_rules(atom, deriv) if any(isinstance(b, Not) for b in r.body))


def _naf_support_ok(target, deriv, cfg):
    if target is None:
        return False
    d = deriv[target]
    return (
        cfg.min_target_depth <= d.depth <= min(cfg.max_target_depth, cfg.max_depth)
        and _naf_rules_in_derivation(target, deriv) >= cfg.min_naf_rules_in_proof
    )


def choose_naf_example(theory, res, cfg, label_counts=None, label_signs=None):
    label_counts = Counter() if label_counts is None else label_counts
    label_signs = Counter() if label_signs is None else label_signs
    labels = ("entailment", "contradiction", "neutral")
    min_count = min(label_counts[x] for x in labels)
    preferred = [x for x in labels if label_counts[x] == min_count]
    random.shuffle(preferred)
    order = preferred + [x for x in labels if x not in preferred]

    derived = [a for a in res.closure if a not in theory.facts and res.derivations[a].depth > 0]

    def balanced(label, pool):
        want = label_signs[(label, True)] <= label_signs[(label, False)]
        same = [hyp for hyp in pool if hyp.sign == want]
        return same or pool

    for label in order:
        if label == "entailment":
            pool = balanced(label, [a for a in derived if _naf_support_ok(a, res.derivations, cfg)])
            if pool:
                hyp = random.choice(pool)
                return label, hyp, res.derivations[hyp], support_atoms(hyp, res.derivations)
        elif label == "contradiction":
            pool = balanced(label, [opposite(a) for a in derived if _naf_support_ok(a, res.derivations, cfg)])
            if pool:
                hyp = random.choice(pool)
                target = opposite(hyp)
                return label, hyp, res.derivations[target], support_atoms(target, res.derivations)
        else:
            live = {pred_key(r.head) for r in participating_rules(res.derivations)}
            pool = [
                a for a in _all_ground_atoms(theory)
                if a not in res.closure and opposite(a) not in res.closure and pred_key(a) in live
            ]
            if pool:
                hyp = random.choice(balanced(label, pool))
                return label, hyp, None, set()
    return None


def pred_words(pred):
    return pred.replace("_", " ")


def tag_word(pred):
    if pred.startswith("p") and pred[1:].isdigit():
        i = int(pred[1:])
        return TAG_WORDS[i] if i < len(TAG_WORDS) else f"tag {i}"
    return pred_words(pred)


def rel_word(pred):
    if pred.startswith("r") and pred[1:].isdigit():
        i = int(pred[1:])
        return REL_WORDS[i] if i < len(REL_WORDS) else f"relation {i}"
    return pred_words(pred)


def binary_text(pred, args, sign=True, pack="surface"):
    x, y = args
    if pack == "abstract" and pred.startswith("r"):
        rel = rel_word(pred)
        return f"{x} is {'not ' if not sign else ''}{rel} to {y}"
    if pred in {"parent", "ancestor", "sibling", "spouse", "aunt_or_uncle"}:
        rel = pred_words(pred)
        art = "an" if rel[0] in "aeiou" else "a"
        return f"{x} is {'not ' if not sign else ''}{art} {rel} of {y}"
    if pred in {"left_of", "right_of", "above", "below", "inside"}:
        return f"{x} is {'not ' if not sign else ''}{pred_words(pred)} {y}"
    if pred == "contains":
        return f"{x} {'does not contain' if not sign else 'contains'} {y}"
    if pred == "disjoint":
        return f"{x} is {'not ' if not sign else ''}disjoint from {y}"
    p = pred_words(pred)
    return f"{x} {p} {y}" if sign else f"{x} does not stand in the {p} relation to {y}"


def atom_text(a, pack="surface"):
    if len(a.args) == 1 and pack == "abstract" and a.pred.startswith("p"):
        t = tag_word(a.pred)
        return f"{a.args[0]} is {'not ' if not a.sign else ''}{t} tagged"
    p = pred_words(a.pred)
    neg = "not " if not a.sign else ""
    if len(a.args) == 1:
        return f"{a.args[0]} is {neg}{p}"
    return binary_text(a.pred, a.args, a.sign, pack)


def group_entity_facts(theory, lines, rate):
    """Sometimes render same-property unary facts as one conjunction."""
    groups = defaultdict(list)
    for i, atom in enumerate(theory.facts):
        if len(atom.args) == 1:
            groups[(atom.pred, atom.sign)].append(i)
    packed, skipped = {}, set()
    for (pred, sign), ids in groups.items():
        if len(ids) < 2 or random.random() >= rate:
            continue
        names = [theory.facts[i].args[0] for i in ids]
        subject = " and ".join(names) if len(names) == 2 else f"{', '.join(names[:-1])}, and {names[-1]}"
        prop = f"{tag_word(pred)} tagged" if theory.domain_pack == "abstract" and pred.startswith("p") else pred_words(pred)
        packed[ids[0]] = f"{subject} are {'not ' if not sign else ''}{prop}."
        skipped.update(ids[1:])
    facts = [packed.get(i, lines[i]) for i in range(len(theory.facts)) if i not in skipped]
    return facts + lines[len(theory.facts):]


def _lit_schema(a, names=None, pack="surface"):
    names = names or {"?x": "x", "?y": "y", "?z": "z", "?p": "p"}
    args = [names.get(x, x) for x in a.args]
    if len(a.args) == 1 and pack == "abstract" and a.pred.startswith("p"):
        return f"{args[0]} is {'not ' if not a.sign else ''}{tag_word(a.pred)} tagged"
    if len(a.args) == 2 and pack == "abstract" and a.pred.startswith("r"):
        return f"{args[0]} is {'not ' if not a.sign else ''}{rel_word(a.pred)} to {args[1]}"
    p = pred_words(a.pred)
    if len(a.args) == 1:
        return f"{args[0]} is {'not ' if not a.sign else ''}{p}"
    return binary_text(a.pred, args, a.sign, pack)


def _not_lit_schema(n, names=None, pack="surface"):
    return "it cannot be shown that " + _lit_schema(n.atom, names, pack)


def _rel_phrase(pred, subj, obj, pack="surface"):
    return _lit_schema(Atom(pred, (subj, obj)), pack=pack)


def rule_text(rule, rid=None, pack="surface"):
    atoms = body_atoms(rule.body)
    nots = [b for b in rule.body if isinstance(b, Not)]
    checks = body_checks(rule.body)
    names = {"?x": "x", "?y": "y", "?z": "z", "?p": "p"}
    symbolic = pack == "abstract"
    parts = [_lit_schema(a, names, pack) for a in atoms]
    parts += [_not_lit_schema(n, names, pack) for n in nots]
    parts += [f"{c[1][1:]} is different from {c[2][1:]}" for c in checks if c[0] == "!="]
    body = " and ".join(parts)
    head = _lit_schema(rule.head, names, pack)
    vars_ = sorted(
        {v[1:] for a in atoms + body_not_atoms(rule.body) + [rule.head] for v in a.args if is_var(v)}
        | {v[1:] for c in checks for v in c[1:] if is_var(v)}
    )
    q = ", ".join(vars_)
    templates = [
        f"For all {q}, if {body}, then {head}.",
        f"Whenever {body}, {head}.",
        f"From {body}, it follows that {head}.",
    ]
    if nots:
        pos_body = " and ".join(_lit_schema(a, names, pack) for a in atoms) or "the condition holds"
        exc_body = " and ".join(_lit_schema(n.atom, names, pack) for n in nots)
        templates += [
            f"By default, if {pos_body}, then {head}, unless it can be shown that {exc_body}.",
            f"If {pos_body}, and it cannot be shown that {exc_body}, then {head}.",
        ]
        if rid is None:
            rid = random.randrange(len(templates))
        return templates[rid % len(templates)], (rule.shape, rid % len(templates), "naf", "default")
    if len(atoms) == 1 and len(atoms[0].args) == 1 and len(rule.head.args) == 1:
        if symbolic:
            a = ("not " if not atoms[0].sign else "") + tag_word(atoms[0].pred)
            h = ("not " if not rule.head.sign else "") + tag_word(rule.head.pred)
            templates += [
                f"Anyone who is {a} tagged is {h} tagged.",
                f"Every {a}-tagged person is {h} tagged.",
                f"If a person is {a} tagged, then that person is {h} tagged.",
            ]
        else:
            a = ("not " if not atoms[0].sign else "") + pred_words(atoms[0].pred)
            h = ("not " if not rule.head.sign else "") + pred_words(rule.head.pred)
            templates += [f"All things that are {a} are {h}.", f"Every {a} entity is {h}.", f"Being {a} implies being {h}."]
    elif len(atoms) == 2 and all(len(a.args) == 1 and a.args == ("?x",) for a in atoms) and rule.head.args == ("?x",):
        a = tag_word(atoms[0].pred) if symbolic else pred_words(atoms[0].pred)
        b = tag_word(atoms[1].pred) if symbolic else pred_words(atoms[1].pred)
        h = ("not " if not rule.head.sign else "") + (tag_word(rule.head.pred) if symbolic else pred_words(rule.head.pred))
        tagged = " tagged" if symbolic else ""
        templates += [
            f"Anyone who is {a}{tagged} and {b}{tagged} is {h}{tagged}.",
            f"Every {a}-tagged person who is {b}{tagged} is {h}{tagged}." if symbolic else f"Every {a} entity that is also {b} is {h}.",
            f"If a person is {a}{tagged} and {b}{tagged}, then that person is {h}{tagged}.",
        ]
    elif len(atoms) == 2 and len(atoms[0].args) == 2 and atoms[1].args == ("?y",) and rule.head.args == ("?x",):
        r = rel_word(atoms[0].pred) if symbolic else pred_words(atoms[0].pred)
        a = tag_word(atoms[1].pred) if symbolic else pred_words(atoms[1].pred)
        h = ("not " if not rule.head.sign else "") + (tag_word(rule.head.pred) if symbolic else pred_words(rule.head.pred))
        tagged = " tagged" if symbolic else ""
        rel = _rel_phrase(atoms[0].pred, "a person", f"a {a}{tagged} person", pack)
        templates += [
            f"Anyone {r} to a {a}{tagged} person is {h}{tagged}." if symbolic else f"If {rel}, then that person is {h}.",
            f"If a person is {r} to someone {a}{tagged}, then that person is {h}{tagged}." if symbolic else f"When {rel}, that person is {h}.",
            f"A person is {h}{tagged} when they are {r} to a {a}{tagged} person." if symbolic else f"A person is {h} when {rel}.",
        ]
    elif len(atoms) == 2 and len(atoms[0].args) == 2 and atoms[1].args == ("?x",) and rule.head.args == ("?y",):
        r = rel_word(atoms[0].pred) if symbolic else pred_words(atoms[0].pred)
        a = tag_word(atoms[1].pred) if symbolic else pred_words(atoms[1].pred)
        h = ("not " if not rule.head.sign else "") + (tag_word(rule.head.pred) if symbolic else pred_words(rule.head.pred))
        tagged = " tagged" if symbolic else ""
        rel = _rel_phrase(atoms[0].pred, f"a {a}{tagged} person", "someone", pack)
        templates += [
            f"Anyone {r} from a {a}{tagged} person is {h}{tagged}." if symbolic else f"If {rel}, then that other person is {h}.",
            f"If a {a}{tagged} person is {r} to someone, then that other person is {h}{tagged}." if symbolic else f"When {rel}, that other person is {h}.",
            f"People reached by a {r} relation from a {a}{tagged} person are {h}{tagged}." if symbolic else f"People reached when {rel} are {h}.",
        ]
    elif len(atoms) == 2 and all(len(a.args) == 2 for a in atoms) and rule.head.args == ("?x", "?z") and rule.head.sign:
        r = rel_word(atoms[0].pred) if symbolic else pred_words(atoms[0].pred)
        s = rel_word(atoms[1].pred) if symbolic else pred_words(atoms[1].pred)
        h = rel_word(rule.head.pred) if symbolic else pred_words(rule.head.pred)
        b1 = _rel_phrase(atoms[0].pred, "one person", "a second person", pack)
        b2 = _rel_phrase(atoms[1].pred, "the second", "a third person", pack)
        hd = _rel_phrase(rule.head.pred, "the first", "the third", pack)
        templates += [
            f"If {b1}, and {b2}, then {hd}.",
            f"When {b1} and {b2}, {hd}.",
            f"{r.title()} relations followed by {s} relations imply {h} relations.",
        ]
    elif len(atoms) == 1 and len(atoms[0].args) == 2 and len(rule.head.args) == 2 and rule.head.args == ("?y", "?x") and rule.head.sign:
        r = rel_word(atoms[0].pred) if symbolic else pred_words(atoms[0].pred)
        h = rel_word(rule.head.pred) if symbolic else pred_words(rule.head.pred)
        body = _rel_phrase(atoms[0].pred, "one person", "another", pack)
        head = _rel_phrase(rule.head.pred, "the second", "the first", pack)
        templates += [
            f"If {body}, then {head}.",
            f"Every {r} relation creates a {h} relation in the reverse direction.",
        ]
    if rid is None:
        rid = random.randrange(len(templates))
    return templates[rid % len(templates)], (rule.shape, rid % len(templates), "if" if rid % 3 == 0 else "whenever", "normal")


def _naf_predicate_text(pred, pack="surface"):
    if pack == "abstract" and pred.startswith("p"):
        return tag_word(pred)
    if pack == "abstract" and pred.startswith("r"):
        return rel_word(pred)
    return "abnormal" if pred == "ab_bird" else pred_words(pred)


def _naf_property_text(atom, pack="surface"):
    pred = _naf_predicate_text(atom.pred, pack)
    if pack == "abstract" and atom.pred.startswith("p"):
        pred = f"{pred}-tagged"
    elif atom.pred in {"bird", "penguin"}:
        pred = f"a {pred}"
    return f"{'not ' if not atom.sign else ''}{pred}"


def _naf_atom_text(atom, pack="surface"):
    arg_text = lambda arg: arg[1:] if is_var(arg) else arg.capitalize()
    if len(atom.args) == 1:
        return f"{arg_text(atom.args[0])} is {_naf_property_text(atom, pack)}"
    x, y = (arg_text(arg) for arg in atom.args)
    if pack == "abstract" and atom.pred.startswith("r"):
        return f"{x} is {'not ' if not atom.sign else ''}{rel_word(atom.pred)} to {y}"
    if atom.pred == "helps":
        return f"{x} {'helps' if atom.sign else 'does not help'} {y}"
    return binary_text(atom.pred, (x, y), atom.sign, pack)


def _english_join(items):
    if len(items) < 2:
        return "".join(items)
    if len(items) == 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _capitalized(text):
    return text[:1].upper() + text[1:]


def _naf_group_text(atom, pack):
    prop = _naf_property_text(atom, pack)
    if atom.pred in {"bird", "penguin"} and atom.sign:
        return f"{_naf_predicate_text(atom.pred, pack).capitalize()}s"
    if atom.sign:
        return f"{_capitalized(prop)} people"
    return f"People who are {prop}"


def _naf_complement_text(atom, pack):
    if atom.pred in {"bird", "penguin"}:
        noun = f"{_naf_predicate_text(atom.pred, pack)}s"
        return f"{'not ' if not atom.sign else ''}{noun}"
    return _naf_property_text(atom, pack)


def _naf_person_text(atom, pack):
    prop = _naf_property_text(atom, pack)
    article = "an" if prop[0] in "aeiou" else "a"
    return prop if atom.pred in {"bird", "penguin"} else f"{article} {prop} person"


def _naf_rule_text(rule, pack="surface"):
    atoms = body_atoms(rule.body)
    exceptions = [item.atom for item in rule.body if isinstance(item, Not)]
    if (
        len(atoms) == 1
        and len(atoms[0].args) == len(rule.head.args) == 1
        and atoms[0].args == rule.head.args
    ):
        text = f"{_naf_group_text(atoms[0], pack)} are {_naf_complement_text(rule.head, pack)}"
        if exceptions:
            text += f" unless {' or '.join(_naf_complement_text(atom, pack) for atom in exceptions)}"
        return text + "."
    if (
        len(atoms) == 2
        and len(atoms[0].args) == 2
        and len(atoms[1].args) == len(rule.head.args) == 1
        and atoms[1].args == (atoms[0].args[1],)
        and rule.head.args == (atoms[0].args[0],)
    ):
        obj = _naf_person_text(atoms[1], pack)
        if pack == "abstract" and atoms[0].pred.startswith("r"):
            subject = f"People {rel_word(atoms[0].pred)} to {obj}"
        elif atoms[0].pred == "helps":
            subject = f"People who help {obj}"
        else:
            subject = f"People for whom {_naf_atom_text(atoms[0], pack)} and {_naf_atom_text(atoms[1], pack)}"
        text = f"{subject} are {_naf_complement_text(rule.head, pack)}"
        if exceptions:
            phrases = [
                _naf_complement_text(atom, pack)
                if atom.args == rule.head.args
                else f"that person is {_naf_property_text(atom, pack)}"
                if atom.args == atoms[1].args
                else _naf_atom_text(atom, pack)
                for atom in exceptions
            ]
            text += f" unless {' or '.join(phrases)}"
        return text + "."
    conditions = [_naf_atom_text(atom, pack) for atom in atoms]
    conditions += [f"{item[1][1:]} differs from {item[2][1:]}" for item in body_checks(rule.body) if item[0] == "!="]
    text = f"If {' and '.join(conditions)}, {_naf_atom_text(rule.head, pack)}"
    if exceptions:
        text += f" unless {' or '.join(_naf_atom_text(atom, pack) for atom in exceptions)}"
    return text + "."


def render_naf(theory):
    """Render a NAF theory compactly without changing its logical representation."""
    unary, other = {}, []
    for atom in theory.facts:
        if len(atom.args) == 1:
            unary.setdefault(atom.args[0], []).append(_naf_property_text(atom, theory.domain_pack))
        else:
            other.append(_capitalized(_naf_atom_text(atom, theory.domain_pack)) + ".")
    facts = "\n".join(
        [f"{entity.capitalize()} is {_english_join(properties)}." for entity, properties in unary.items()] + other
    )
    rules = "\n".join(_naf_rule_text(rule, theory.domain_pack) for rule in theory.rules)
    return facts, rules


def render(theory):
    lines, source = [], {}
    for a in theory.facts:
        source[a] = len(lines)
        lines.append(atom_text(a, theory.domain_pack) + ".")
    surf = []
    for r in theory.rules:
        txt, key = rule_text(r, pack=theory.domain_pack)
        source[r] = len(lines)
        surf.append((*key, theory.domain_pack))
        lines.append(txt)
    return lines, source, surf


def trace_for(target, deriv, source, pack="surface"):
    if target is None:
        return ""
    lines, seen = [], {}

    def rec(a):
        if a in seen:
            return seen[a]
        d = deriv[a]
        parent_ids = [rec(p) for p in d.parents]
        idx = len(lines) + 1
        seen[a] = idx
        if d.rule is None:
            ctx = f"P{source.get(a, '?')}"
        else:
            ctx = ";".join([f"P{source.get(d.rule, '?')}"] + [str(i) for i in parent_ids])
        lines.append(f"{idx}. [{ctx}] {atom_text(a, pack)}")
        return idx

    rec(target)
    return "\n".join(lines)


def derivation_rules(atom, deriv):
    if atom is None:
        return set()
    d = deriv[atom]
    if d.rule is None:
        return set()
    out = {d.rule}
    for p in d.parents:
        out |= derivation_rules(p, deriv)
    return out


def participating_rules(deriv):
    return {d.rule for d in deriv.values() if d.rule is not None}


def bin_key(label, hyp, d, support, theory, target=None, deriv=None):
    depth = 0 if d is None else d.depth
    support_size = len(support)
    uses_binary = any(len(a.args) == 2 for a in support | {hyp})
    rules = derivation_rules(target, deriv) if target is not None and deriv is not None else set()
    uses_signed = any(not r.head.sign for r in rules)
    uses_comp = any("composition" in r.shape for r in rules)
    return (label, min(depth, 4), min(support_size, 4), uses_binary, uses_signed, uses_comp, theory.domain_pack)


def target_for(label, hyp):
    return opposite(hyp) if label == "contradiction" else hyp if label == "entailment" else None


def indexed_premise(lines):
    return "\n".join(f"[{i}] {x}" for i, x in enumerate(lines))


def format_candidate_list(candidates, pack):
    return "\n".join(f"[{i}] {atom_text(a, pack)}." for i, a in enumerate(candidates))


def case_metadata(case, key=None):
    d = case.derivation
    return edict(
        premise=case.lines,
        hypothesis=atom_text(case.hyp, case.theory.domain_pack) + ".",
        label=case.label,
        domain_pack=case.theory.domain_pack,
        depth=None if d is None else d.depth,
        hypothesis_sign=case.hyp.sign,
        support_indices=[] if case.target is None else sorted(support_sources(case.target, case.res.derivations, case.source)),
        live_rule_rate=case.live_rule_rate,
        proof_rule_rate=case.proof_rule_rate,
        bin_key=repr(key) if key is not None else "",
        surface_bins=[repr(x) for x in case.surf],
        neutral_completion=(
            "" if case.neutral_completion is None
            else atom_text(case.neutral_completion, case.theory.domain_pack) + "."
        ),
        cot=trace_for(case.target, case.res.derivations, case.source, case.theory.domain_pack),
    )


def naf_case_metadata(case, key=None):
    meta = case_metadata(case, key)
    target = case.target
    meta.naf_rule_count = sum(1 for r in case.theory.rules if any(isinstance(b, Not) for b in r.body))
    meta.naf_rules_in_proof = 0 if target is None else _naf_rules_in_derivation(target, case.res.derivations)
    meta.strata = {repr(k): v for k, v in (stratify_rules(case.theory.rules) or {}).items()}
    return meta


def shallow_closure(theory, keep_shapes):
    rules = [r for r in theory.rules if r.shape in keep_shapes]
    return chase(Theory(
        theory.facts, rules, theory.denials,
        theory.pred_sigs, theory.entities, theory.domain_pack
    ))


def reject_shortcut(case):
    if case.target is None:
        return False
    for shapes in ({"u_imp"}, {"u_imp", "converse"}, {"u_imp", "u_and"}):
        res = shallow_closure(case.theory, shapes)
        if case.target in res.closure or opposite(case.target) in res.closure:
            return True
    return False


def _hard_neutral(theory, res, cfg, label_signs):
    targets = [
        atom for atom in res.closure
        if atom not in theory.facts
        and opposite(atom) not in theory.facts
        and res.derivations[atom].depth > 0
        and hard_target(atom, res.derivations, cfg)
    ]
    candidates = [(hyp, target) for target in targets for hyp in (target, opposite(target))]
    wanted_sign = label_signs[("neutral", True)] <= label_signs[("neutral", False)]
    random.shuffle(candidates)
    candidates.sort(key=lambda pair: pair[0].sign != wanted_sign)
    for hyp, target in candidates:
        proof_size = len(support_atoms(target, res.derivations)) + len(derivation_rules(target, res.derivations))
        focused = _focused_theory(theory, res, [target], cfg.n_distractors + 1)
        missing_facts = list(support_atoms(target, res.derivations))
        random.shuffle(missing_facts)
        for missing in missing_facts:
            reduced = Theory(
                [atom for atom in focused.facts if atom != missing],
                focused.rules, focused.denials, focused.pred_sigs,
                focused.entities, focused.domain_pack,
            )
            reduced_res = chase(reduced, max_depth=None)
            mentioned = {arg for atom in reduced.facts for arg in atom.args}
            if (
                not reduced_res.inconsistent
                and _label_for_atom(reduced_res, hyp) == "neutral"
                and all(arg in mentioned for arg in hyp.args)
                and len(reduced.facts) + len(reduced.rules) == proof_size + cfg.n_distractors
            ):
                completed = close_with(reduced, [missing])
                if target in completed.derivations and hard_target(target, completed.derivations, cfg):
                    return reduced, reduced_res, hyp, missing
    return None


def generate_case(cfg, allowed_labels=("entailment", "contradiction", "neutral"), state=None):
    if state is None:
        state = {}
    bins = state.setdefault("bins", Counter())
    label_counts = state.setdefault("label_counts", Counter())
    label_signs = state.setdefault("label_signs", Counter())
    label_depths = state.setdefault("label_depths", Counter())
    surface = state.setdefault("surface", Counter())
    allowed_labels = set(allowed_labels)
    for _ in range(500):
        theory = sample_theory(cfg)
        res = chase(theory, max_depth=None)
        if res.inconsistent:
            continue
        choice = choose_example(theory, res, cfg, label_signs, label_depths)
        if not choice:
            continue
        label, hyp, derivation, support = choice
        if label not in allowed_labels:
            continue
        neutral_completion = None
        if label == "neutral":
            hard_neutral = _hard_neutral(theory, res, cfg, label_signs)
            if hard_neutral is None:
                continue
            theory, res, hyp, neutral_completion = hard_neutral
            derivation, support = None, set()
        ls_key = (label, hyp.sign)
        other = label_signs[(label, not hyp.sign)]
        if label_signs[ls_key] > other + 6 and random.random() > 0.2:
            continue
        lines, source, surf = render(theory)
        if surf and surface:
            ordered = sorted(surface.values())
            median = ordered[len(ordered) // 2]
            overused = any(surface[x] > max(4, median * 2) for x in surf)
            if overused and random.random() < 0.85:
                continue
        target = target_for(label, hyp)
        d = None if target is None else res.derivations[target]
        used_rules = derivation_rules(target, res.derivations) if target is not None else set()
        live_rule_rate = len(participating_rules(res.derivations)) / max(1, len(theory.rules))
        proof_rule_rate = len(used_rules) / max(1, len(theory.rules))
        if live_rule_rate < 0.25:
            continue
        if label != "neutral" and proof_rule_rate < 0.10:
            continue
        key = bin_key(label, hyp, d, support, theory, target, res.derivations)
        case = MultistepCase(
            theory, res, label, hyp, target, d, support, lines, source, surf,
            live_rule_rate, proof_rule_rate, neutral_completion,
        )
        if reject_shortcut(case):
            continue
        focused = (
            theory if target is None else
            _focused_theory(theory, res, [target], cfg.n_distractors)
        )
        focused_res = chase(focused, max_depth=None)
        if focused_res.inconsistent or _label_for_atom(focused_res, hyp) != label:
            continue
        theory, res = focused, focused_res
        lines, source, surf = render(theory)
        d = None if target is None else res.derivations[target]
        support = set() if target is None else support_atoms(target, res.derivations)
        used_rules = derivation_rules(target, res.derivations) if target is not None else set()
        live_rule_rate = len(participating_rules(res.derivations)) / max(1, len(theory.rules))
        proof_rule_rate = len(used_rules) / max(1, len(theory.rules))
        case = MultistepCase(
            theory, res, label, hyp, target, d, support, lines, source, surf,
            live_rule_rate, proof_rule_rate, neutral_completion,
        )
        min_label = min([label_counts[x] for x in allowed_labels] or [0])
        label_needs_examples = label_counts[label] <= min_label + 3
        if bins[key] >= cfg.max_bin_size and not label_needs_examples and random.random() > 0.2:
            continue
        bins[key] += 1
        label_counts[label] += 1
        label_signs[ls_key] += 1
        if d is not None:
            label_depths[(label, d.depth)] += 1
        surface.update(surf)
        return case, key
    return None, None


def generate_naf_case(cfg, allowed_labels=("entailment", "contradiction", "neutral"), state=None):
    if state is None:
        state = {}
    label_counts = state.setdefault("label_counts", Counter())
    label_signs = state.setdefault("label_signs", Counter())
    allowed_labels = set(allowed_labels)
    for _ in range(500):
        theory = sample_naf_theory(cfg)
        try:
            res = naf_chase(theory, max_depth=None)
        except ValueError:
            continue
        if res.inconsistent:
            continue
        choice = choose_naf_example(theory, res, cfg, label_counts, label_signs)
        if not choice:
            continue
        label, hyp, derivation, support = choice
        if label not in allowed_labels:
            continue
        lines, source, surf = render(theory)
        target = target_for(label, hyp)
        used_rules = derivation_rules(target, res.derivations) if target is not None else set()
        live_rule_rate = len(participating_rules(res.derivations)) / max(1, len(theory.rules))
        proof_rule_rate = len(used_rules) / max(1, len(theory.rules))
        key = bin_key(label, hyp, derivation, support, theory, target, res.derivations)
        label_counts[label] += 1
        label_signs[(label, hyp.sign)] += 1
        return MultistepCase(theory, res, label, hyp, target, derivation, support, lines, source, surf, live_rule_rate, proof_rule_rate), key
    return None, None


def _without_source(theory, source, idx):
    return Theory(
        [a for a in theory.facts if source.get(a) != idx],
        [r for r in theory.rules if source.get(r) != idx],
        theory.denials,
        theory.pred_sigs,
        theory.entities,
        theory.domain_pack,
    )


def _with_extra_fact(theory, fact):
    return Theory(
        list(theory.facts) + [fact],
        theory.rules,
        theory.denials,
        theory.pred_sigs,
        theory.entities,
        theory.domain_pack,
    )


def _naf_label_after(theory, hyp):
    res = naf_chase(theory, max_depth=None)
    if res.inconsistent:
        return None
    return _label_for_atom(res, hyp)


def make_naf_removal_flip_case(cfg):
    targets = ("entailment", "contradiction", "neutral")
    for _ in range(500):
        case, key = generate_naf_case(cfg)
        if not case:
            continue
        lines, source = case.lines, case.source
        current = case.label
        target_label = random.choice([x for x in targets if x != current])
        sols = []
        exception_sols = []
        for idx in range(len(lines)):
            try:
                label = _naf_label_after(_without_source(case.theory, source, idx), case.hyp)
            except ValueError:
                continue
            if label == target_label:
                sols.append((idx,))
                obj = next((x for x, j in source.items() if j == idx), None)
                if isinstance(obj, Atom) and obj.pred in {"flagged", "p1", "blocked", "p3", "penguin", "p6"}:
                    exception_sols.append((idx,))
        chosen = exception_sols or sols
        if chosen:
            return edict(
                case=case,
                key=key,
                target_label=target_label,
                answer=chosen[0],
                valid_supports=[list(x) for x in chosen],
            )
    return None


def make_naf_addition_flip_case(cfg):
    targets = ("entailment", "contradiction", "neutral")
    for _ in range(500):
        case, key = generate_naf_case(cfg)
        if not case:
            continue
        current = case.label
        target_label = random.choice([x for x in targets if x != current])
        candidates = []
        for a in _all_ground_atoms(case.theory):
            if a in case.theory.facts or opposite(a) in case.theory.facts:
                continue
            try:
                label = _naf_label_after(_with_extra_fact(case.theory, a), case.hyp)
            except ValueError:
                continue
            if label == target_label:
                candidates.insert(0, a)
            elif len(candidates) < cfg.n_distractors + 3:
                candidates.append(a)
            if len(candidates) >= max(4, cfg.n_distractors + 3) and any(
                _naf_label_after(_with_extra_fact(case.theory, c), case.hyp) == target_label for c in candidates[:3]
            ):
                break
        good = []
        for i, cand in enumerate(candidates):
            if _naf_label_after(_with_extra_fact(case.theory, cand), case.hyp) == target_label:
                good.append(i)
        if good:
            order = list(range(len(candidates[: max(4, cfg.n_distractors + 3)])))
            random.shuffle(order)
            cands = [candidates[i] for i in order]
            good = [i for i, old in enumerate(order) if _naf_label_after(_with_extra_fact(case.theory, candidates[old]), case.hyp) == target_label]
            return edict(
                case=case,
                key=key,
                candidates=cands,
                target_label=target_label,
                answer=(good[0],),
                valid_supports=[[i] for i in good],
            )
    return None


def near_miss_abducibles(theory, removed, n):
    out, seen = [], set(removed)
    seen |= set(theory.facts)
    ents = list(next(iter(theory.entities.values())))
    unaries = [s.name for s in theory.pred_sigs.values() if len(s.arg_types) == 1]
    binaries = [s.name for s in theory.pred_sigs.values() if len(s.arg_types) == 2]
    def add(a):
        if a in seen:
            return
        seen.add(a)
        out.append(a)
    for a in removed:
        add(opposite(a))
        if len(a.args) == 1:
            others = [e for e in ents if e != a.args[0]]
            preds = [p for p in unaries if p != a.pred]
            if others:
                add(Atom(a.pred, (random.choice(others),), a.sign))
            if preds:
                add(Atom(random.choice(preds), a.args, a.sign))
        elif len(a.args) == 2:
            preds = [p for p in binaries if p != a.pred]
            others = [e for e in ents if e not in a.args] or ents
            add(Atom(a.pred, (a.args[1], a.args[0]), a.sign))
            if preds:
                add(Atom(random.choice(preds), a.args, a.sign))
            add(Atom(a.pred, (a.args[0], random.choice(others)), a.sign))
    random.shuffle(out)
    return out[:n]


def minimal_abductions(theory, candidates, target, max_k=2):
    for k in range(1, max_k + 1):
        out = []
        for ids in itertools.combinations(range(len(candidates)), k):
            res = close_with(theory, [candidates[i] for i in ids])
            if not res.inconsistent and target in res.closure:
                out.append(ids)
        if out:
            return out
    return []


def necessary_indices(case):
    if case.target is None:
        return []
    # Any globally necessary premise must occur in every proof, including this one.
    support = sorted(support_sources(case.target, case.res.derivations, case.source))
    nec = []
    for i in support:
        keep = lambda x: case.source.get(x) != i
        sub = Theory(
            [a for a in case.theory.facts if keep(a)],
            [r for r in case.theory.rules if keep(r)],
            case.theory.denials,
            case.theory.pred_sigs,
            case.theory.entities,
            case.theory.domain_pack,
        )
        if case.target not in chase(sub, max_depth=None).closure:
            nec.append(i)
    return nec


def parse_indices(s):
    return set(parse_space_ints(s) or [])


def _entails(theory, res, atom):
    return atom in res.closure and opposite(atom) not in res.closure


def _query_words(pred, pack):
    if pack == "abstract" and pred.startswith("p"):
        return f"{tag_word(pred)} tagged"
    return pred_words(pred)


def _binary_query(pred, x, count, pack):
    label = "How many other entities" if count else "Which other entities"
    if pack == "abstract" and pred.startswith("r"):
        return f"{label} can {x} be shown to be {rel_word(pred)} to?"
    if pred in {"parent", "ancestor", "sibling", "spouse", "aunt_or_uncle"}:
        rel = pred_words(pred)
        art = "an" if rel[0] in "aeiou" else "a"
        return f"{label} can {x} be shown to be {art} {rel} of?"
    if pred in {"left_of", "right_of", "above", "below", "inside"}:
        return f"{label} can {x} be shown to be {pred_words(pred)}?"
    if pred == "contains":
        return f"{label} can {x} be shown to contain?"
    if pred == "disjoint":
        return f"{label} can {x} be shown to be disjoint from?"
    return f"{label} can {x} be shown to stand in the {pred_words(pred)} relation to?"


def _join_answer(xs):
    return ", ".join(sorted(xs))


def _hard_answer_atoms(atoms, res, cfg):
    # Set-valued QA already requires closing the theory for every entity. Requiring
    # each answer to also have many independent leaves makes deep planted chains
    # disappear at high levels, so use proof depth as its hardness criterion.
    min_depth = max(2, int(cfg.min_target_depth) - 1)
    return [
        a for a in atoms
        if a in res.derivations
        and min_depth <= res.derivations[a].depth <= cfg.max_depth
    ]


def _world_query(theory, res, source, cfg):
    mode = cfg.answer_mode
    modes = ("count", "list") if mode == "any" else (mode,)
    random_modes = random.sample(modes, len(modes))
    sigs = list(theory.pred_sigs.values())
    random.shuffle(sigs)
    allow_zero = random.random() < getattr(cfg, "zero_answer_rate", 0.0)

    for qmode in random_modes:
        for sig in sigs:
            if len(sig.arg_types) == 1:
                atoms = [Atom(sig.name, (x,)) for x in theory.entities[sig.arg_types[0]]]
                answer_atoms = [a for a in atoms if _entails(theory, res, a)]
                if not answer_atoms:
                    if not allow_zero:
                        continue
                    hard = []
                else:
                    hard = _hard_answer_atoms(answer_atoms, res, cfg)
                    if not hard:
                        continue
                answers = sorted(a.args[0] for a in answer_atoms)
                pred = _query_words(sig.name, theory.domain_pack)
                question = f"Which entities can be shown to be {pred}?" if qmode == "list" else f"How many entities can be shown to be {pred}?"
                answer = "none" if qmode == "list" and not answers else _join_answer(answers)
                answer = "0" if qmode == "count" and not answers else str(len(answers)) if qmode == "count" else answer
                support = sorted(set().union(*(support_sources(a, res.derivations, source) for a in answer_atoms)))
                return edict(mode=qmode, question=question, answer=answer, support_indices=support, atoms=answer_atoms, hard_atoms=hard)

            if len(sig.arg_types) == 2:
                xs, ys = (theory.entities[t] for t in sig.arg_types)
                x = random.choice(tuple(xs))
                atoms = [Atom(sig.name, (x, y)) for y in ys if y != x]
                answer_atoms = [a for a in atoms if _entails(theory, res, a)]
                if not answer_atoms:
                    if not allow_zero:
                        continue
                    hard = []
                else:
                    hard = _hard_answer_atoms(answer_atoms, res, cfg)
                    if not hard:
                        continue
                answers = sorted(a.args[1] for a in answer_atoms)
                question = _binary_query(sig.name, x, qmode == "count", theory.domain_pack)
                answer = "none" if qmode == "list" and not answers else _join_answer(answers)
                answer = "0" if qmode == "count" and not answers else str(len(answers)) if qmode == "count" else answer
                support = sorted(set().union(*(support_sources(a, res.derivations, source) for a in answer_atoms)))
                return edict(mode=qmode, question=question, answer=answer, support_indices=support, atoms=answer_atoms, hard_atoms=hard)
    return None


def make_direct_abduction_case(cfg):
    entities, sigs, _, _ = _domain_pack("surface", cfg)
    es = list(next(iter(entities.values())))
    x, y = random.sample(es, 2)
    unaries = [s.name for s in sigs.values() if len(s.arg_types) == 1]
    binaries = [s.name for s in sigs.values() if len(s.arg_types) == 2]
    label = random.choice(["entailment", "contradiction"])
    shapes = ["unary", "bridge", "conjunctive"]
    if cfg.max_abduction_size >= 2:
        shapes.append("two_missing_conjunctive")
    shape = random.choice(shapes)
    facts, rules, removed = [], [], []

    if shape == "bridge" and binaries and len(unaries) >= 3:
        r = random.choice(binaries)
        a, b, c = random.sample(unaries, 3)
        removed = [Atom(a, (y,))]
        facts = [Atom(r, (x, y)), Atom(random.choice(unaries), (random.choice(es),))]
        rules = [
            Rule((Atom(r, ("?x", "?y")), Atom(a, ("?y",))), Atom(b, ("?x",)), "abduction", "rel_y"),
            Rule((Atom(b, ("?x",)),), Atom(c, ("?x",), label == "entailment"), "abduction", "u_imp"),
        ]
        hyp = Atom(c, (x,), True)
    elif shape == "conjunctive" and len(unaries) >= 4:
        a, b, c, d = random.sample(unaries, 4)
        removed = [Atom(a, (x,))]
        facts = [Atom(b, (x,)), Atom(a, (y,)), Atom(random.choice(unaries), (random.choice(es),))]
        rules = [
            Rule((Atom(a, ("?x",)), Atom(b, ("?x",))), Atom(c, ("?x",)), "abduction", "u_and"),
            Rule((Atom(c, ("?x",)),), Atom(d, ("?x",), label == "entailment"), "abduction", "u_imp"),
        ]
        hyp = Atom(d, (x,), True)
    elif shape == "two_missing_conjunctive" and len(unaries) >= 4:
        a, b, c, d = random.sample(unaries, 4)
        removed = [Atom(a, (x,)), Atom(b, (x,))]
        filler_preds = [p for p in unaries if p not in {a, b, c, d}] or unaries
        facts = [Atom(a, (y,)), Atom(b, (y,)), Atom(random.choice(filler_preds), (random.choice(es),))]
        rules = [
            Rule((Atom(a, ("?x",)), Atom(b, ("?x",))), Atom(c, ("?x",)), "abduction", "u_and"),
            Rule((Atom(c, ("?x",)),), Atom(d, ("?x",), label == "entailment"), "abduction", "u_imp"),
        ]
        hyp = Atom(d, (x,), True)
    else:
        a, b, c = random.sample(unaries, 3)
        removed = [Atom(a, (x,))]
        facts = [Atom(a, (y,)), Atom(random.choice(unaries), (random.choice(es),))]
        rules = [
            Rule((Atom(a, ("?x",)),), Atom(b, ("?x",)), "abduction", "u_imp"),
            Rule((Atom(b, ("?x",)),), Atom(c, ("?x",), label == "entailment"), "abduction", "u_imp"),
        ]
        hyp = Atom(c, (x,), True)

    target = target_for(label, hyp)
    clean_facts, seen = [], set()
    for fact in facts:
        if fact not in seen and opposite(fact) not in seen:
            clean_facts.append(fact)
            seen.add(fact)
    facts = clean_facts
    theory = Theory(facts, rules, [], sigs, entities, "surface")
    used = set(theory.facts)
    for _ in range(max(0, int(cfg.n_distractors) - 1)):
        fact = _fresh_fact(theory, used)
        if fact:
            theory.facts.append(fact)
    if target in chase(theory, None).closure:
        return None
    if target not in close_with(theory, removed).closure:
        return None

    candidates = removed + near_miss_abducibles(theory, removed, cfg.n_candidates * 3)
    filtered = []
    for cand in candidates:
        if cand in filtered:
            continue
        cres = close_with(theory, [cand])
        if cand in removed or (not cres.inconsistent and target not in cres.closure):
            filtered.append(cand)
        if len(filtered) >= cfg.n_candidates:
            break
    used = set(filtered) | set(theory.facts)
    while len(filtered) < cfg.n_candidates:
        cand = _fresh_fact(theory, used)
        if not cand:
            break
        cres = close_with(theory, [cand])
        if not cres.inconsistent and target not in cres.closure:
            filtered.append(cand)
    random.shuffle(filtered)
    sols = minimal_abductions(theory, filtered, target, cfg.max_abduction_size)
    if not sols or (cfg.require_unique and len(sols) != 1):
        return None
    lines, _, _ = render(theory)
    return edict(theory=theory, label=label, hyp=hyp, target=target, lines=lines,
                 candidates=filtered, answer=list(sols[0]), domain_pack="surface")


class MultistepNLI(Task):
    summary = "Multi-hop natural language inference over chained logic facts and rules."
    def __init__(self, config=None):
        super().__init__(config=config or MultistepNLIConfig())
        self.balancing_key_ratio = 1 / 3
        self._case_state = {}

    def generate_entry(self):
        for _ in range(3):
            case, key = generate_case(self.config, state=self._case_state)
            if not case:
                continue
            meta = case_metadata(case, key)
            premise = group_entity_facts(case.theory, meta.premise, self.config.group_entities)
            meta.payload = {"premise": "\n".join(premise), "hypothesis": meta.hypothesis}
            mapping = {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe"}
            return Entry(meta, mapping[case.label])
        raise RuntimeError("could not generate a consistent multistep_nli example")

    def render_prompt(self, meta):
        return (
            f"{render_payload(meta.payload)}\n\n"
            "Is the hypothesis true given the premise? "
            "The answer is Yes, No, or Maybe."
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip().lower().rstrip(".") == str(entry.answer).strip().lower())

    def balancing_key(self, problem):
        return problem.answer


class DefeasibleNLI(Task):
    summary = "NLI using defeasible logic rules and negation as failure."

    def __init__(self, config=None):
        super().__init__(config=config or DefeasibleNLIConfig())
        self.balancing_key_ratio = 1 / 3
        self._case_state = {}

    def generate_entry(self):
        case, key = generate_naf_case(self.config, state=self._case_state)
        if case:
            meta = naf_case_metadata(case, key)
            facts, rules = render_naf(case.theory)
            meta.payload = {
                "facts": facts,
                "rules": rules,
                "hypothesis": _capitalized(_naf_atom_text(case.hyp, case.theory.domain_pack)) + ".",
            }
            mapping = {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe"}
            return Entry(meta, mapping[case.label])
        raise RuntimeError("could not generate a stratified_naf_nli example")

    def render_prompt(self, meta):
        return (
            "An `unless` condition must be shown to block its rule.\n\n"
            f"{render_payload(meta.payload)}\n\n"
            "Is the hypothesis true? Answer Yes, No, or Maybe."
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip().lower().rstrip(".") == str(entry.answer).strip().lower())

    def balancing_key(self, problem):
        return problem.answer

class MultistepEvidenceRetrieval(Task):
    summary = "Retrieve the specific premise indexes required to prove a logical hypothesis."
    def __init__(self, config=None):
        super().__init__(config=config or MultistepNLIConfig())
        self._case_state = {}

    def generate_entry(self):
        for _ in range(200):
            case, key = generate_case(self.config, ("entailment", "contradiction"), self._case_state)
            if not case:
                continue
            nec = necessary_indices(case)
            if not nec:
                continue
            meta = case_metadata(case, key)
            meta.necessary_indices = nec
            meta.valid_supports = [nec]
            meta.support_indices = nec
            meta.payload = {"premise": indexed_premise(meta.premise), "hypothesis": meta.hypothesis}
            answer = " ".join(str(i) for i in nec)
            return Entry(meta, answer)
        raise RuntimeError("could not generate a unique-support multistep_evidence_retrieval example")

    def render_prompt(self, meta):
        verb = "entail" if meta.label == "entailment" else "contradict"
        return (
            f"{render_payload(meta.payload)}\n\n"
            f"Which premise statements are necessary to {verb} the hypothesis, "
            "meaning removing any one of them breaks that result?\n"
            "Answer with space-separated indexes."
        )

    def score_answer(self, answer, entry):
        pred = parse_indices(answer)
        valid = [set(x) for x in entry.metadata.get("valid_supports", [])]
        return float(any(pred == x for x in valid))


class MultistepAbduction(Task):
    summary = "Find the missing facts from candidates to satisfy a target hypothesis."
    def __init__(self, config=None):
        super().__init__(config=config or MultistepAbductionConfig())
        self._case_state = {}

    def generate_entry(self):
        for _ in range(500):
            abd = make_direct_abduction_case(self.config)
            if not abd:
                continue
            meta = edict(
                premise=abd.lines,
                hypothesis=atom_text(abd.hyp, abd.domain_pack) + ".",
                candidates=[atom_text(a, abd.domain_pack) + "." for a in abd.candidates],
                label=abd.label,
                domain_pack=abd.domain_pack,
            )
            meta.payload = {
                "premise": indexed_premise(meta.premise),
                "hypothesis": meta.hypothesis,
                "candidate_facts": indexed_premise(meta.candidates),
            }
            answer = " ".join(str(i) for i in abd.answer)
            return Entry(meta, answer)
        raise RuntimeError("could not generate a consistent multistep_abduction example")

    def render_prompt(self, meta):
        mode = "entail the hypothesis" if meta.label == "entailment" else "contradict the hypothesis"
        return (
            f"{render_payload(meta.payload)}\n\n"
            f"Which smallest set of candidate facts, if added to the premise, make the premise {mode}?\n"
            "Do not include candidate facts that are not needed.\n"
            "Answer with space-separated indexes."
        )

    def score_answer(self, answer, entry):
        return score_space_ints(answer, entry)


class LogicQA(Task):
    summary = "Answer multi-step logical reasoning queries over rule-based theories."
    def __init__(self, config=None):
        super().__init__(config=config or LogicQAConfig())
        self.balancing_key_ratio = 1 / 3

    def generate_entry(self):
        for _ in range(300):
            theory = sample_theory(self.config)
            res = chase(theory, max_depth=None)
            if res.inconsistent:
                continue
            lines, source, _ = render(theory)
            query = _world_query(theory, res, source, self.config)
            if not query:
                continue
            if query.atoms:
                theory = _focused_theory(
                    theory, res, query.atoms, self.config.n_distractors
                )
                res = chase(theory, max_depth=None)
                lines, source, _ = render(theory)
                if any(not _entails(theory, res, atom) for atom in query.atoms):
                    continue
            meta = edict(
                premise=lines,
                question=query.question,
                answer_mode=query.mode,
                domain_pack=theory.domain_pack,
                support_indices=query.support_indices,
                max_answer_depth=max((res.derivations[a].depth for a in query.atoms), default=0),
                hard_answer_depths=[res.derivations[a].depth for a in query.hard_atoms],
                cot="\n\n".join(trace_for(a, res.derivations, source, theory.domain_pack) for a in query.atoms if a in res.derivations),
            )
            meta.payload = {"premise": "\n".join(lines), "question": query.question}
            return Entry(meta, query.answer)
        raise RuntimeError("could not generate a logic_qa example")

    def render_prompt(self, meta):
        if meta.answer_mode == "count":
            fmt = "Answer with one integer."
        else:
            fmt = "Answer with names in alphabetical order, comma-separated, or 'none'."
        return f"{render_payload(meta.payload)}\n\n{fmt}"

    def score_answer(self, answer, entry):
        metadata = entry.metadata if hasattr(entry, "metadata") else entry["metadata"]
        reference = entry.answer if hasattr(entry, "answer") else entry["answer"]
        if metadata["answer_mode"] == "count":
            try:
                text = str(answer).strip().casefold().rstrip(".")
                pred = 0 if text in {"", "none", "no one", "nobody", "[]"} else int(text)
                return float(pred == int(reference))
            except ValueError:
                return 0.0
        try:
            import ast
            text = str(answer).strip()
            if text.casefold().rstrip(".") in {"", "none", "no one", "nobody", "[]"}:
                pred = []
            else:
                try:
                    pred = ast.literal_eval(text)
                    if not isinstance(pred, (list, tuple)):
                        pred = [str(pred)]
                except Exception:
                    pred = [x.strip() for x in text.split(",") if x.strip()]
                    if len(pred) == 1 and pred[0].casefold().rstrip(".") == "none":
                        pred = []
            ref_text = str(reference).strip()
            truth = [] if ref_text.casefold().rstrip(".") in {"", "none", "[]"} else [x.strip() for x in ref_text.split(",") if x.strip()]
            return float(sorted(str(x).strip().casefold() for x in pred) == sorted(x.casefold() for x in truth))
        except Exception:
            return 0.0

    def balancing_key(self, problem):
        return (problem.metadata.domain_pack, problem.metadata.answer_mode)
