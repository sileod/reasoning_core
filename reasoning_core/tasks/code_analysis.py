import ast
import hashlib
import random
from collections import deque
from dataclasses import dataclass
from itertools import product
from textwrap import indent

from gramforge import generate, init_grammar
from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround


@dataclass
class CodeAnalysisConfig(Config):
    n_vars: int = 2
    domain_size: int = 2
    n_modes: int = 3
    n_predicates: int = 3
    formula_depth: int = 2
    branchiness: int = 2
    max_states: int = 32
    min_witness_len: int = 2
    max_retries: int = 200

    def apply_difficulty(self, level):
        self.n_vars = sround(self.n_vars + 0.45 * level)
        self.domain_size = sround(self.domain_size + 0.2 * level)
        self.n_modes = sround(self.n_modes + 0.45 * level)
        self.n_predicates = sround(self.n_predicates + 0.5 * level)
        self.formula_depth = sround(self.formula_depth + 0.55 * level)
        self.branchiness = sround(self.branchiness + 0.25 * level)
        self.max_states = sround(self.max_states + 20 * level)


NAME_POOLS = {
    "cat": ("mode", "phase", "status", "stage", "color"),
    "int": ("x", "y", "count", "level", "index"),
    "bool": ("flag", "ready", "locked", "active", "valid"),
}
LABEL_POOLS = (
    ("idle", "wait", "done", "fail", "retry", "hold", "check", "exit"),
    ("red", "blue", "green", "amber", "white", "black", "cyan", "gold"),
    ("new", "open", "busy", "closed", "stale", "ready", "paused", "final"),
)


class _BadProgram(Exception):
    pass


class _NeedChoice(Exception):
    def __init__(self, options):
        self.options = list(options)


def _state_name(i):
    return f"s{i}"


def _formula(op, *args):
    return (op,) + args


def _depth(f):
    if f[0] == "atom":
        return 1
    if f[0] in ("!", "EX", "AX", "EF", "AF", "EG", "AG"):
        return 1 + _depth(f[1])
    return 1 + max(_depth(f[1]), _depth(f[2]))


def _var_kind(domain):
    if set(domain) == {"False", "True"}:
        return "bool"
    if all(value.isdigit() for value in domain):
        return "int"
    return "cat"


def _pick_variables(cfg):
    n_vars = max(2, min(5, int(cfg.n_vars)))
    used = set()
    vars_ = []
    kinds = [random.choice(("cat", "int"))]
    kinds += [random.choice(("cat", "int", "bool")) for _ in range(n_vars - 1)]
    for kind in kinds:
        names = [name for name in NAME_POOLS[kind] if name not in used]
        name = random.choice(names)
        used.add(name)
        if kind == "cat":
            size = max(2, min(int(cfg.n_modes), len(LABEL_POOLS[0])))
            domain = list(random.choice(LABEL_POOLS)[:size])
        elif kind == "int":
            domain = list(map(str, range(max(2, int(cfg.domain_size)))))
        else:
            domain = ["False", "True"]
        random.shuffle(domain)
        vars_.append((name, tuple(domain)))
    while _domain_size(vars_) > cfg.max_states and len(vars_) > 2:
        vars_.pop()
    return vars_


def _condition(vars_, compound=True):
    if random.random() < 0.12:
        return "random.choice([False, True])"
    name, domain = random.choice(vars_)
    kind = _var_kind(domain)
    if kind == "bool":
        atom = name if random.random() < 0.5 else f"not {name}"
    elif kind == "int" and random.random() < 0.45:
        values = sorted(map(int, domain))
        op = random.choice(("<", ">="))
        atom = f"{name} {op} {random.choice(values[1:])}"
    else:
        atom = f"{name} {random.choice(('==', '!='))} {_literal(random.choice(domain))}"
    if compound and random.random() < 0.3:
        other = _condition(vars_, compound=False)
        atom = f"({atom}) {random.choice(('and', 'or'))} ({other})"
    return atom


def _rhs(name, domain, vars_):
    kind = _var_kind(domain)
    if kind == "cat":
        offset = random.randrange(1, len(domain))
        shuffled = list(domain[offset:] + domain[:offset])
        pairs = ", ".join(f"{_literal(a)}: {_literal(b)}" for a, b in zip(domain, shuffled))
        table = "{" + pairs + "}"
        choices = "[" + ", ".join(map(_literal, domain)) + "]"
        return random.choice((
            _literal(random.choice(domain)),
            f"random.choice({choices})",
            f"{table}[{name}]",
            f"{_literal(random.choice(domain))} if {_condition(vars_)} else {name}",
        ))
    if kind == "bool":
        return random.choice((
            f"not {name}",
            "random.choice([False, True])",
            f"bool({_condition(vars_)})",
            f"{name} if {_condition(vars_)} else not {name}",
        ))
    size = len(domain)
    other_ints = [
        other for other, values in vars_
        if _var_kind(values) == "int" and other != name
    ]
    other = random.choice(other_ints) if other_ints else name
    step = random.randint(1, max(1, size - 1))
    return random.choice((
        f"({name} + {step}) % {size}",
        f"({name} + {other} + {step}) % {size}",
        f"min({name} + {step}, {size - 1})",
        f"max({name} - {step}, 0)",
        f"random.choice({list(map(int, domain))})",
        f"{random.choice(domain)} if {_condition(vars_)} else {name}",
    ))


def _assignment(vars_, allow_tuple=True):
    if allow_tuple and len(vars_) > 1 and random.random() < 0.25:
        chosen = random.sample(vars_, 2)
        lhs = ", ".join(name for name, _ in chosen)
        rhs = ", ".join(_rhs(name, domain, vars_) for name, domain in chosen)
        return f"{lhs} = {rhs}\n"
    name, domain = random.choice(vars_)
    return f"{name} = {_rhs(name, domain, vars_)}\n"


def _match_statement(vars_):
    name, domain = random.choice(vars_)
    values = random.sample(list(domain), min(2, len(domain)))
    lines = [f"match {name}:"]
    for value in values:
        lines += [f"    case {_literal(value)}:", indent(_assignment(vars_), "        ").rstrip()]
    lines += ["    case _:", indent(_assignment(vars_), "        ").rstrip()]
    return "\n".join(lines) + "\n"


def _transition_grammar(vars_):
    rules = init_grammar(["py"])

    def render_if(cond, yes, no=None):
        text = f"if {cond.render('py')}:\n{indent(yes.render('py'), '    ')}"
        if no is not None:
            text += f"else:\n{indent(no.render('py'), '    ')}"
        return text

    def render_guard(cond, action):
        return f"if {cond.render('py')}:\n{indent(action.render('py'), '    ')}    return\n"

    def render_if_chain(first, yes, second, middle, no):
        return (
            f"if {first.render('py')}:\n{indent(yes.render('py'), '    ')}"
            f"elif {second.render('py')}:\n{indent(middle.render('py'), '    ')}"
            f"else:\n{indent(no.render('py'), '    ')}"
        )

    rules("CTX", "")
    rules("ACTION(CTX)", lambda _: _assignment(vars_))
    rules("MATCH(CTX)", lambda _: _match_statement(vars_))
    rules("COND(CTX)", lambda _: _condition(vars_))
    rules("IF(COND, BLOCK)", render_if)
    rules("IF(COND, BLOCK, BLOCK)", render_if, weight=1.5)
    rules("IF(COND, BLOCK, COND, BLOCK, BLOCK)", render_if_chain, weight=0.6)
    rules("GUARD(COND, ACTION)", render_guard, weight=0.8)
    rules("STMT(ACTION)", "0", weight=2.0)
    rules("STMT(MATCH)", "0", weight=0.5)
    rules("STMT(IF)", "0", weight=1.2)
    rules("STMT(GUARD)", "0", weight=0.8)
    rules("BLOCK(STMT)", "0")
    rules("BLOCK(STMT, STMT)", lambda a, b: a.render("py") + b.render("py"), weight=1.2)

    def render_program(body):
        names = ", ".join(name for name, _ in vars_)
        initial = ", ".join(_literal(domain[0]) for _, domain in vars_)
        block = indent(body.render("py"), "    ")
        return f"import random\n\n{names} = {initial}\n\ndef step():\n    global {names}\n{block}"

    rules("PROGRAM(BLOCK)", render_program)
    rules("start(PROGRAM)", "0")
    return rules


class CodeAnalysis(Task):
    summary = "Analyze toy finite-state Python-like programs with CTL temporal formulas."
    config_cls = CodeAnalysisConfig

    def __init__(self, config=None):
        super().__init__(config=config)
        self.balancing_key_ratio = 0.35

    def generate_entry(self):
        cfg = self.config
        modes = ["holds", "states", "rank", "witness"]
        random.shuffle(modes)

        for query_type in modes:
            for _ in range(max(1, cfg.max_retries)):
                try:
                    kripke = self._make_kripke()
                except _BadProgram:
                    continue
                if len(kripke.reachable) < min(3, len(kripke.states)):
                    continue
                predicates = self._make_predicates(kripke)
                if not predicates:
                    continue
                formula = self._make_formula_for_query(query_type, kripke, predicates)
                solved = self._solve_query(query_type, formula, kripke, predicates)
                if solved is None:
                    continue
                answer, metrics = solved
                if self._accept(query_type, answer, metrics, kripke):
                    metadata = self._metadata(query_type, formula, kripke, predicates, metrics)
                    return Entry(metadata=metadata, answer=answer)

        raise RuntimeError(f"Failed to generate code_analysis task. Config: {cfg}")

    def render_prompt(self, m):
        parts = [
            "Program:", "```python", m.program, "```", "",
            "Start from the assignments above; each transition calls `step()`.",
        ]
        if "random.choice" in m.program:
            parts += ["", "Each `random.choice` outcome is a nondeterministic transition."]
        parts += ["", f"Formula: {m.formula}", f"Property: {m.property_text}", ""]
        if m.query_type == "holds":
            parts.append("Does the property hold from the initial state? Answer Yes or No.")
        elif m.query_type == "states":
            parts.append(f"State tuples use ({m.state_variables}).")
            parts.append("Which reachable valuations satisfy it? Answer as a sorted list of tuples.")
        elif m.query_type == "rank":
            parts.append("When does the initial state first enter the least fixed point?")
            parts.append("Iteration 0 contains states satisfying the inner condition. Answer with an integer or never.")
        else:
            if m.witness_kind == "counterexample":
                parts.append("Give the shortest lexicographic valuation path showing failure from the initial state.")
            else:
                parts.append("Give the shortest lexicographic valuation path reaching the inner condition.")
            parts.append(f"State tuples use ({m.state_variables}). Answer as a list of tuples.")
        return "\n".join(parts)

    def score_answer(self, answer, entry):
        metadata = entry["metadata"] if isinstance(entry, dict) else entry.metadata
        q = metadata["query_type"]
        ref = entry["answer"] if isinstance(entry, dict) else entry.answer
        if q == "holds":
            return float(_norm_yesno(answer) == ref)
        if q == "states":
            return float(set(_parse_valuations(answer)) == set(_parse_valuations(ref)))
        if q == "rank":
            return float(str(answer).strip().lower().rstrip(".") == ref)
        if q == "witness":
            return float(_parse_valuations(answer) == _parse_valuations(ref))
        return 0.0

    def balancing_key(self, problem):
        m = problem.metadata
        if m.query_type == "holds":
            return f"holds:{problem.answer}"
        return f"{m.query_type}:{m.difficulty_bucket}"

    def deduplication_key(self, problem):
        m = problem.metadata
        raw = "|".join([m.query_type, m.formula, m.program])
        return hashlib.sha1(raw.encode()).hexdigest()

    def _make_kripke(self):
        cfg = self.config
        vars_ = _pick_variables(cfg)
        states = list(product(*[range(len(domain)) for _, domain in vars_]))
        index = {state: i for i, state in enumerate(states)}
        depth = 6 + min(4, int(cfg.branchiness))
        program = generate(_transition_grammar(vars_), depth=depth, min_depth=4) @ "py"
        compile(program, "<code_analysis_program>", "exec")
        if len(program.splitlines()) > 32:
            raise _BadProgram("program too long")
        succ = []
        for state in states:
            choices = [index[s] for s in self._execute_successors(program, vars_, state)]
            succ.append(sorted(set(choices)) or [index[state]])

        reachable = _reachable(succ, 0)
        features = sorted({
            type(node).__name__.lower()
            for node in ast.walk(ast.parse(program))
            if isinstance(node, (ast.If, ast.Match, ast.Return, ast.Tuple, ast.Dict, ast.IfExp))
        })
        return edict(
            vars=vars_, states=states, succ=succ, initial=0, reachable=reachable,
            program=program, syntax="+".join(features or ["assign"]),
        )

    def _execute_successors(self, program, vars_, state):
        pending = [()]
        outcomes = set()
        while pending:
            path = pending.pop()
            namespace = {}
            exec(program, namespace, namespace)
            for i, (name, domain) in enumerate(vars_):
                namespace[name] = _runtime_value(domain[state[i]])
            choice_index = 0

            def choice(options):
                nonlocal choice_index
                if choice_index < len(path):
                    value = path[choice_index]
                    choice_index += 1
                    return value
                raise _NeedChoice(options)

            namespace["random"] = edict(choice=choice)
            try:
                namespace["step"]()
            except _NeedChoice as needed:
                if len(pending) + len(outcomes) > 128:
                    raise _BadProgram("too many nondeterministic paths")
                pending.extend(path + (value,) for value in needed.options)
                continue
            try:
                outcome = tuple(
                    domain.index(_domain_value(namespace[name]))
                    for name, domain in vars_
                )
            except (KeyError, ValueError, TypeError) as error:
                raise _BadProgram("transition escaped its finite domain") from error
            outcomes.add(outcome)
        return outcomes

    def _make_predicates(self, k):
        candidates = []
        for vi, (name, domain) in enumerate(k.vars):
            for value_i, value in enumerate(domain):
                sat = {sid for sid, state in enumerate(k.states) if state[vi] == value_i}
                expr = f"{name} == {_literal(value)}"
                candidates.append((expr, sat))
            if all(v.isdigit() for v in domain) and len(domain) > 2:
                threshold = max(map(int, domain))
                sat = {
                    sid for sid, state in enumerate(k.states)
                    if int(domain[state[vi]]) < threshold
                }
                candidates.append((f"{name} < {threshold}", sat))
        random.shuffle(candidates)
        preds = []
        seen = set()
        for expr, sat in candidates:
            key = tuple(sorted(sat))
            reachable_sat = sat & k.reachable
            if key in seen or len(reachable_sat) in (0, len(k.reachable)):
                continue
            seen.add(key)
            preds.append((f"p{len(preds)}", expr, sat))
            if len(preds) >= self.config.n_predicates:
                break
        return preds

    def _make_formula_for_query(self, query_type, k, preds):
        if query_type == "holds":
            inner = self._random_formula(k, preds, max(1, self.config.formula_depth - 1), temporal=True)
            return _formula(random.choice(("EX", "AX", "EF", "AF", "EG", "AG")), inner)
        if query_type == "rank":
            inner = self._random_formula(k, preds, max(1, self.config.formula_depth - 1), temporal=False)
            return _formula(random.choice(("EF", "AF")), inner)
        if query_type == "witness":
            inner = self._random_formula(k, preds, max(1, self.config.formula_depth - 1), temporal=False)
            return _formula(random.choice(("EF", "AG")), inner)
        return self._random_formula(k, preds, max(1, self.config.formula_depth), temporal=True)

    def _random_formula(self, k, preds, depth, temporal=True):
        if depth <= 1 or random.random() < 0.25:
            return _formula("atom", random.choice(preds)[0])
        if temporal and random.random() < 0.55:
            return _formula(random.choice(("EX", "AX", "EF", "AF", "EG", "AG")), self._random_formula(k, preds, depth - 1))
        if random.random() < 0.2:
            return _formula("!", self._random_formula(k, preds, depth - 1, temporal))
        return _formula(
            random.choice(("&", "|", "->")),
            self._random_formula(k, preds, depth - 1, temporal),
            self._random_formula(k, preds, depth - 1, temporal),
        )

    def _solve_query(self, query_type, formula, k, preds):
        pred_map = {name: sat for name, _, sat in preds}
        sat, fp_iters = self._sat(formula, k.succ, pred_map)
        reachable_sat = sat & k.reachable
        if query_type == "holds":
            return ("Yes" if k.initial in sat else "No"), self._metrics(k, formula, reachable_sat, fp_iters, pred_map)
        if query_type == "states":
            valuations = sorted(_state_valuation(k, sid) for sid in reachable_sat)
            answer = repr(valuations)
            return answer, self._metrics(k, formula, reachable_sat, fp_iters, pred_map)
        if query_type == "rank":
            rank = self._entry_rank(formula, k.succ, pred_map, k.initial)
            answer = "never" if rank is None else str(rank)
            metrics = self._metrics(k, formula, reachable_sat, fp_iters, pred_map)
            metrics.fp_iterations = -1 if rank is None else rank
            return answer, metrics
        path = self._witness_path(formula, k, pred_map)
        if path is None:
            return None
        metrics = self._metrics(k, formula, reachable_sat, fp_iters, pred_map)
        metrics.shortest_witness_len = len(path) - 1
        return repr([_state_valuation(k, sid) for sid in path]), metrics

    def _sat(self, f, succ, preds):
        n = len(succ)
        all_states = set(range(n))
        op = f[0]
        if op == "atom":
            return set(preds[f[1]]), 0
        if op == "!":
            s, it = self._sat(f[1], succ, preds)
            return all_states - s, it
        if op in ("&", "|", "->"):
            a, ia = self._sat(f[1], succ, preds)
            b, ib = self._sat(f[2], succ, preds)
            if op == "&":
                return a & b, max(ia, ib)
            if op == "|":
                return a | b, max(ia, ib)
            return (all_states - a) | b, max(ia, ib)

        inner, inner_i = self._sat(f[1], succ, preds)
        if op == "EX":
            return {i for i, ss in enumerate(succ) if any(j in inner for j in ss)}, inner_i
        if op == "AX":
            return {i for i, ss in enumerate(succ) if all(j in inner for j in ss)}, inner_i
        if op == "EF":
            return _least_fixed_point(inner, succ, existential=True, base=inner_i)
        if op == "AF":
            return _least_fixed_point(inner, succ, existential=False, base=inner_i)
        if op == "EG":
            return _greatest_fixed_point(inner, succ, existential=True, base=inner_i)
        if op == "AG":
            return _greatest_fixed_point(inner, succ, existential=False, base=inner_i)
        raise ValueError(op)

    def _entry_rank(self, formula, succ, preds, init):
        op = formula[0]
        if op not in ("EF", "AF"):
            return None
        inner, _ = self._sat(formula[1], succ, preds)
        current = set(inner)
        if init in current:
            return 0
        for iteration in range(1, len(succ) + 1):
            if op == "EF":
                add = {i for i, ss in enumerate(succ) if any(j in current for j in ss)}
            else:
                add = {i for i, ss in enumerate(succ) if all(j in current for j in ss)}
            new = current | add
            if init in new:
                return iteration
            if new == current:
                return None
            current = new
        return None

    def _witness_path(self, formula, k, preds):
        op = formula[0]
        if op == "EF":
            target, _ = self._sat(formula[1], k.succ, preds)
        elif op == "AG":
            good, _ = self._sat(formula[1], k.succ, preds)
            target = set(range(len(k.states))) - good
        else:
            return None
        return _shortest_lex_path(k.succ, k.initial, target, key=lambda sid: _state_valuation(k, sid))

    def _metrics(self, k, formula, sat, fp_iters, preds):
        reachable = k.reachable
        branch_counts = [len(k.succ[i]) for i in reachable]
        sat_fraction = len(sat) / max(1, len(k.reachable))
        effort, mixed, inner_count = self._temporal_effort(formula, k, preds, sat)
        return edict(
            n_states=len(k.states),
            n_edges=sum(len(s) for s in k.succ),
            reachable_states=len(reachable),
            formula_depth=_depth(formula),
            temporal_depth=_temporal_depth(formula),
            fp_iterations=fp_iters,
            shortest_witness_len=0,
            sat_count=len(sat),
            sat_fraction=round(sat_fraction, 3),
            branching_entropy=round(sum(branch_counts) / max(1, len(branch_counts)), 3),
            root_operator=formula[0],
            temporal_effort=effort,
            mixed_initial_branches=mixed,
            inner_count=inner_count,
        )

    def _temporal_effort(self, formula, k, preds, sat):
        op = formula[0]
        if op not in ("EX", "AX", "EF", "AF", "EG", "AG"):
            return 0, False, 0
        inner, _ = self._sat(formula[1], k.succ, preds)
        inner_count = len(inner)
        initial = k.initial
        successors = set(k.succ[initial])
        mixed = bool(successors & inner) and bool(successors - inner)
        if op in ("EX", "AX"):
            return len(successors), mixed, inner_count
        if op == "EF":
            if initial in sat:
                return _shortest_distance(k.succ, initial, inner), False, inner_count
            return _max_shortest_distance(k.succ, initial), False, inner_count
        if op == "AF":
            if initial in sat:
                return self._entry_rank(formula, k.succ, preds, initial), False, inner_count
            effort = _shortest_reachable_cycle(k.succ, initial, set(range(len(k.succ))) - inner)
            return effort, False, inner_count
        if op == "EG":
            if initial in sat:
                return _shortest_reachable_cycle(k.succ, initial, inner), False, inner_count
            return _eg_removal_rank(inner, k.succ, initial), False, inner_count
        if initial in sat:  # AG
            return _max_shortest_distance(k.succ, initial), False, inner_count
        effort = _shortest_distance(k.succ, initial, set(range(len(k.succ))) - inner)
        return effort, False, inner_count

    def _accept(self, query_type, answer, metrics, k):
        if metrics.reachable_states < min(3, metrics.n_states):
            return False
        if query_type == "states" and metrics.sat_count in (0, metrics.reachable_states):
            return False
        if query_type == "states" and metrics.temporal_depth == 0:
            return random.random() < 0.2
        if query_type == "rank":
            if answer == "never":
                required = max(1, int(self.config.min_witness_len))
                return (
                    metrics.temporal_effort is not None
                    and metrics.temporal_effort >= required
                    and random.random() < 0.25
                )
            return int(answer) >= max(1, int(self.config.min_witness_len))
        if query_type == "witness":
            return metrics.shortest_witness_len >= max(1, int(self.config.min_witness_len))
        if query_type == "holds":
            required = 2 if metrics.root_operator in {"EX", "AX"} else max(1, int(self.config.min_witness_len))
            if metrics.inner_count in (0, metrics.n_states):
                return False
            if metrics.temporal_effort is None or metrics.temporal_effort < required:
                return False
            if (metrics.root_operator, answer) in {("EX", "Yes"), ("AX", "No")}:
                return metrics.mixed_initial_branches
            return True
        return True

    def _metadata(self, query_type, formula, k, preds, metrics):
        formula_text = self._render_formula(formula)
        pred_text = "\n".join(f"{name} := {expr}" for name, expr, _ in preds)
        program = k.program
        if "random.choice" not in program:
            program = program.removeprefix("import random\n\n")
        compile(program, "<code_analysis_program>", "exec")
        bucket = f"d{metrics.formula_depth}:s{metrics.n_states}:t{metrics.temporal_depth}"
        return edict(
            program=program,
            predicates=pred_text,
            formula=formula_text,
            property_text=self._render_property(formula, {name: expr for name, expr, _ in preds}),
            query_type=query_type,
            witness_kind="counterexample" if query_type == "witness" and formula[0] == "AG" else "witness",
            initial_state=_state_name(k.initial),
            state_variables=", ".join(name for name, _ in k.vars),
            answer_format=query_type,
            syntax=k.syntax,
            difficulty_bucket=bucket,
            n_states=metrics.n_states,
            n_edges=metrics.n_edges,
            reachable_states=metrics.reachable_states,
            formula_depth=metrics.formula_depth,
            temporal_depth=metrics.temporal_depth,
            fp_iterations=metrics.fp_iterations,
            shortest_witness_len=metrics.shortest_witness_len,
            sat_count=metrics.sat_count,
            sat_fraction=metrics.sat_fraction,
            branching_entropy=metrics.branching_entropy,
            root_operator=metrics.root_operator,
            temporal_effort=metrics.temporal_effort,
            mixed_initial_branches=metrics.mixed_initial_branches,
            inner_count=metrics.inner_count,
        )

    def _render_formula(self, f):
        op = f[0]
        if op == "atom":
            return f[1]
        if op == "!":
            return f"!({self._render_formula(f[1])})"
        if op in ("EX", "AX", "EF", "AF", "EG", "AG"):
            return f"{op}({self._render_formula(f[1])})"
        return f"({self._render_formula(f[1])} {op} {self._render_formula(f[2])})"

    def _render_property(self, f, predicates=None):
        op = f[0]
        if op == "atom":
            return predicates.get(f[1], f[1]) if predicates else f[1]
        if op == "!":
            return f"not ({self._render_property(f[1], predicates)})"
        if op == "&":
            return f"({self._render_property(f[1], predicates)}) and ({self._render_property(f[2], predicates)})"
        if op == "|":
            return f"({self._render_property(f[1], predicates)}) or ({self._render_property(f[2], predicates)})"
        if op == "->":
            return f"if {self._render_property(f[1], predicates)}, then {self._render_property(f[2], predicates)}"
        if op == "EX":
            return f"after some next step, ({self._render_property(f[1], predicates)})"
        if op == "AX":
            return f"after every next step, ({self._render_property(f[1], predicates)})"
        if op == "EF":
            return f"on some execution, eventually ({self._render_property(f[1], predicates)})"
        if op == "AF":
            return f"on every execution, eventually ({self._render_property(f[1], predicates)})"
        if op == "EG":
            return f"on some infinite execution, always ({self._render_property(f[1], predicates)})"
        if op == "AG":
            return f"at every reachable state, ({self._render_property(f[1], predicates)})"
        return self._render_formula(f)


def _domain_size(vars_):
    size = 1
    for _, domain in vars_:
        size *= len(domain)
    return size


def _literal(value):
    if value in ("True", "False") or value.isdigit():
        return value
    return repr(value)


def _runtime_value(value):
    if value in ("True", "False"):
        return value == "True"
    if value.isdigit():
        return int(value)
    return value


def _domain_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _state_valuation(k, sid):
    state = k.states[sid]
    return tuple(_runtime_value(domain[state[i]]) for i, (_, domain) in enumerate(k.vars))


def _least_fixed_point(base_set, succ, existential, base=0):
    current = set(base_set)
    for iteration in range(1, len(succ) + 1):
        if existential:
            add = {i for i, ss in enumerate(succ) if any(j in current for j in ss)}
        else:
            add = {i for i, ss in enumerate(succ) if all(j in current for j in ss)}
        new = current | add
        if new == current:
            return current, base + iteration - 1
        current = new
    return current, base + len(succ)


def _greatest_fixed_point(base_set, succ, existential, base=0):
    current = set(base_set)
    for iteration in range(1, len(succ) + 1):
        if existential:
            new = {i for i in current if any(j in current for j in succ[i])}
        else:
            new = {i for i in current if all(j in current for j in succ[i])}
        if new == current:
            return current, base + iteration - 1
        current = new
    return current, base + len(succ)


def _reachable(succ, start):
    seen = {start}
    q = deque([start])
    while q:
        i = q.popleft()
        for j in succ[i]:
            if j not in seen:
                seen.add(j)
                q.append(j)
    return seen


def _shortest_lex_path(succ, start, targets, key=None):
    key = key or (lambda node: node)
    if start in targets:
        return [start]
    q = deque([(start, [start])])
    seen = {start}
    while q:
        node, path = q.popleft()
        for nxt in sorted(succ[node], key=key):
            if nxt in seen:
                continue
            npath = path + [nxt]
            if nxt in targets:
                return npath
            seen.add(nxt)
            q.append((nxt, npath))
    return None


def _distances(succ, start, allowed=None):
    allowed = set(range(len(succ))) if allowed is None else allowed
    if start not in allowed:
        return {}
    distances = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in succ[node]:
            if nxt in allowed and nxt not in distances:
                distances[nxt] = distances[node] + 1
                q.append(nxt)
    return distances


def _shortest_distance(succ, start, targets):
    distances = _distances(succ, start)
    found = [distances[target] for target in targets if target in distances]
    return min(found) if found else None


def _max_shortest_distance(succ, start):
    return max(_distances(succ, start).values())


def _shortest_reachable_cycle(succ, start, allowed):
    lengths = []
    for root in _distances(succ, start, allowed):
        for nxt in succ[root]:
            back = _distances(succ, nxt, allowed).get(root)
            if back is not None:
                lengths.append(back + 1)
    return min(lengths) if lengths else None


def _eg_removal_rank(base_set, succ, start):
    current = set(base_set)
    if start not in current:
        return 0
    for iteration in range(1, len(succ) + 1):
        new = {i for i in current if any(j in current for j in succ[i])}
        if start not in new:
            return iteration
        if new == current:
            return None
        current = new
    return None


def _temporal_depth(f):
    if f[0] == "atom":
        return 0
    if f[0] in ("EX", "AX", "EF", "AF", "EG", "AG"):
        return 1 + _temporal_depth(f[1])
    if f[0] == "!":
        return _temporal_depth(f[1])
    return max(_temporal_depth(f[1]), _temporal_depth(f[2]))


def _norm_yesno(answer):
    text = str(answer).strip().lower().rstrip(".")
    if text in {"yes", "y", "true"}:
        return "Yes"
    if text in {"no", "n", "false"}:
        return "No"
    return text


def _parse_valuations(answer):
    text = str(answer).strip()
    candidates = [text] + list(reversed(text.splitlines()))
    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate.strip().strip("`"))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, list) and all(isinstance(item, (tuple, list)) for item in value):
            return [tuple(item) for item in value]
    return []
