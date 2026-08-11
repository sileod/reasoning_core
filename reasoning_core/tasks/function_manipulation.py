import ast
import math
import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from reasoning_core.template import Config, Entry, Task, edict
from reasoning_core.template import stochastic_rounding as sround


X = sp.Symbol("x")
NAMES = iter(())
POLY_OPS = {"var", "poly", "call", "add", "sub", "mul", "diff", "integrate"}
ADVANCED_OPS = {"call", "reciprocal", "exp0", "sin0", "log1p0"}


def _q(x):
    return x if isinstance(x, Fraction) else Fraction(x)


def _fmt(q):
    q = _q(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _small_fraction(bound, denominators=(1, 1, 1, 2, 3)):
    return Fraction(random.randint(-bound, bound), random.choice(denominators))


def _zeros(n):
    return [Fraction(0) for _ in range(n)]


def _cut(a, n):
    return (list(a) + _zeros(n))[:n]


def _add(a, b, n):
    return [x + y for x, y in zip(_cut(a, n), _cut(b, n))]


def _sub(a, b, n):
    return [x - y for x, y in zip(_cut(a, n), _cut(b, n))]


def _mul(a, b, n):
    a, b = _cut(a, n), _cut(b, n)
    return [sum((a[j] * b[i - j] for j in range(i + 1)), Fraction(0))
            for i in range(n)]


def _diff(a, n):
    a = _cut(a, n + 1)
    return _cut([(i + 1) * a[i + 1] for i in range(n)], n)


def _integrate(a, n):
    a = _cut(a, n)
    return [Fraction(0)] + [a[i] / (i + 1) for i in range(n - 1)]


def _reciprocal(a, n):
    a = _cut(a, n)
    if not a[0]:
        raise ZeroDivisionError
    out = _zeros(n)
    out[0] = Fraction(1) / a[0]
    for i in range(1, n):
        out[i] = -sum((a[j] * out[i - j] for j in range(1, i + 1)), Fraction(0)) / a[0]
    return out


def _powers(a, n):
    out = [[Fraction(1)] + _zeros(n - 1)]
    for _ in range(1, n):
        out.append(_mul(out[-1], a, n))
    return out


def _compose(coeffs, arg, center, n):
    delta = _cut(arg, n)
    delta[0] -= center
    if delta[0]:
        raise ValueError("composition center mismatch")
    out = _zeros(n)
    for c, power in zip(_cut(coeffs, n), _powers(delta, n)):
        out = _add(out, [c * x for x in power], n)
    return out


def _exp0(a, n):
    a = _cut(a, n)
    if a[0]:
        raise ValueError("exp0 expects zero constant term")
    out = _zeros(n)
    out[0] = Fraction(1)
    for i in range(1, n):
        out[i] = sum((k * a[k] * out[i - k] for k in range(1, i + 1)), Fraction(0)) / i
    return out


def _sin_cos0(a, n):
    a = _cut(a, n)
    if a[0]:
        raise ValueError("sin0 expects zero constant term")
    sin, cos = _zeros(n), _zeros(n)
    cos[0] = Fraction(1)
    for i in range(1, n):
        sin[i] = sum((k * a[k] * cos[i - k] for k in range(1, i + 1)), Fraction(0)) / i
        cos[i] = -sum((k * a[k] * sin[i - k] for k in range(1, i + 1)), Fraction(0)) / i
    return sin, cos


def _log1p0(a, n):
    a = _cut(a, n)
    if a[0]:
        raise ValueError("log1p0 expects zero constant term")
    one_plus = list(a)
    one_plus[0] = 1
    return _integrate(_mul(_diff(a, n), _reciprocal(one_plus, n), n), n)


def _revert(coeffs, input_center, n):
    coeffs = _cut(coeffs, n)
    if not coeffs[1]:
        raise ValueError("inverse requires a nonzero first derivative")
    u = _zeros(n)
    u[1] = Fraction(1) / coeffs[1]
    delta_coeffs = list(coeffs)
    delta_coeffs[0] = 0
    for degree in range(2, n):
        probe = _compose(delta_coeffs, u, Fraction(0), n)
        u[degree] = -probe[degree] / coeffs[1]
    u[0] = input_center
    return u


def _poly_node(coeffs, center):
    return {"op": "poly", "coeffs": [_fmt(c) for c in coeffs], "center": _fmt(center)}


def _node(op, *args, **data):
    return {"op": op, "args": list(args), **data}


def _new_names():
    for name in "f g p q r s u v w F G P Q R S U V W".split():
        yield name
    i = 0
    while True:
        yield f"f_{i}"
        i += 1


def _function_coeffs(definition, definitions, prec, cache):
    name = definition["name"]
    if name in cache:
        return cache[name]
    if definition["kind"] == "explicit":
        out = [_q(c) for c in definition["coeffs"]]
    else:
        base = definitions[definition["base"]]
        base_coeffs = _function_coeffs(base, definitions, prec, cache)
        out = _revert(base_coeffs, _q(base["input_center"]), prec)
    cache[name] = _cut(out, prec)
    return cache[name]


def _eval(node, definitions, root_center, prec, cache=None):
    cache = {} if cache is None else cache
    op = node["op"]
    if op == "var":
        return [root_center, Fraction(1)] + _zeros(prec - 2)
    if op == "poly":
        if _q(node["center"]) != root_center:
            raise ValueError("polynomial center mismatch")
        return _cut([_q(c) for c in node["coeffs"]], prec)

    args = [_eval(arg, definitions, root_center, prec, cache) for arg in node["args"]]
    if op == "add":
        return _add(args[0], args[1], prec)
    if op == "sub":
        return _sub(args[0], args[1], prec)
    if op == "mul":
        return _mul(args[0], args[1], prec)
    if op == "diff":
        return _diff(args[0], prec)
    if op == "integrate":
        return _integrate(args[0], prec)
    if op == "reciprocal":
        return _reciprocal(args[0], prec)
    if op == "exp0":
        shifted = list(args[0])
        shifted[0] -= _q(node["shift"])
        return _exp0(shifted, prec)
    if op == "sin0":
        shifted = list(args[0])
        shifted[0] -= _q(node["shift"])
        return _sin_cos0(shifted, prec)[0]
    if op == "log1p0":
        shifted = list(args[0])
        shifted[0] -= _q(node["shift"])
        return _log1p0(shifted, prec)
    if op == "call":
        definition = definitions[node["name"]]
        coeffs = _function_coeffs(definition, definitions, prec, cache)
        return _compose(coeffs, args[0], _q(definition["input_center"]), prec)
    raise ValueError(op)


def _explicit_definition(name, input_center, output_center, cfg, invertible=False):
    degree = random.randint(1, cfg.function_degree)
    coeffs = [output_center]
    for _ in range(degree):
        coeffs.append(_small_fraction(cfg.coeff_bound))
    if invertible:
        while not coeffs[1]:
            coeffs[1] = _small_fraction(cfg.coeff_bound)
    elif not any(coeffs[1:]):
        coeffs[1] = Fraction(1)
    return {
        "name": name,
        "kind": "explicit",
        "input_center": _fmt(input_center),
        "output_center": _fmt(output_center),
        "coeffs": [_fmt(c) for c in coeffs],
    }


def _inverse_definition(name, base):
    return {
        "name": name,
        "kind": "inverse",
        "base": base["name"],
        "input_center": base["output_center"],
        "output_center": base["input_center"],
    }


def _branch(center, cfg, kind):
    degree = random.randint(1, cfg.branch_degree)
    coeffs = [_small_fraction(cfg.coeff_bound) for _ in range(degree + 1)]
    coeffs[0] = Fraction(0) if kind == "zero" else Fraction(1)
    if not any(coeffs[1:]):
        coeffs[random.randint(1, degree)] = random.choice([Fraction(-1), Fraction(1)])
    return _poly_node(coeffs, center)


def _ops(node):
    if node["op"] in {"var", "poly"}:
        return []
    return [node["op"]] + [op for arg in node["args"] for op in _ops(arg)]


def _nodes(node):
    return 1 + sum(_nodes(arg) for arg in node.get("args", []))


def _depth(node):
    return 0 if not node.get("args") else 1 + max(_depth(arg) for arg in node["args"])


def _top_call(node):
    return node.get("name") if node["op"] == "call" else None


def _inverse_pair(a, b, definitions):
    if not a or not b:
        return False
    da, db = definitions[a], definitions[b]
    return ((da["kind"] == "inverse" and da["base"] == b) or
            (db["kind"] == "inverse" and db["base"] == a))


def _candidate(current, current_jet, definitions, name_source, root_center, cfg):
    c0 = current_jet[0]
    reusable = [name for name, d in definitions.items()
                if _q(d["input_center"]) == c0]
    room = cfg.max_definitions - len(definitions)

    weighted = [
        ("add", 1.0),
        ("sub", .7),
        ("mul", .9),
        ("diff", cfg.calculus_weight),
        ("integrate", .8 * cfg.calculus_weight),
        ("reciprocal", .35 * cfg.nonlinear_weight if c0 else 0),
        ("exp0", .35 * cfg.nonlinear_weight),
        ("sin0", .35 * cfg.nonlinear_weight),
        ("log1p0", .25 * cfg.nonlinear_weight),
        ("compose", cfg.composition_weight if room >= 1 else 0),
        ("inverse", cfg.inverse_weight if room >= 2 else 0),
        ("reuse", cfg.reuse_weight if reusable else 0),
    ]
    weighted = [(name, weight) for name, weight in weighted if weight > 0]
    op = random.choices([x for x, _ in weighted], [w for _, w in weighted])[0]

    if op == "add":
        return _node("add", current, _branch(root_center, cfg, "zero")), []
    if op == "sub":
        return _node("sub", current, _branch(root_center, cfg, "zero")), []
    if op == "mul":
        return _node("mul", current, _branch(root_center, cfg, "unit")), []
    if op in {"diff", "integrate", "reciprocal"}:
        if ((op == "diff" and current["op"] == "integrate") or
                (op == "integrate" and current["op"] == "diff")):
            if random.random() >= cfg.identity_weight:
                return None, []
        return _node(op, current), []
    if op in {"exp0", "sin0", "log1p0"}:
        return _node(op, current, shift=_fmt(c0)), []
    if op == "compose":
        name = next(name_source)
        output = _small_fraction(cfg.center_bound)
        definition = _explicit_definition(name, c0, output, cfg)
        return _node("call", current, name=name), [definition]
    if op == "inverse":
        base_name, inverse_name = next(name_source), next(name_source)
        input_center = _small_fraction(cfg.center_bound)
        base = _explicit_definition(base_name, input_center, c0, cfg, invertible=True)
        inverse = _inverse_definition(inverse_name, base)
        return _node("call", current, name=inverse_name), [base, inverse]
    if op == "reuse":
        name = random.choice(reusable)
        if _inverse_pair(name, _top_call(current), definitions):
            if random.random() >= cfg.identity_weight:
                return None, []
        return _node("call", current, name=name), []
    raise ValueError(op)


def _definition_degree(definition):
    return len(definition.get("coeffs", [])) - 1


def _reasoning_cost(node, definitions, terminal_order):
    ops = Counter(_ops(node))
    inverse_defs = sum(d["kind"] == "inverse" for d in definitions.values())
    definition_degree = sum(max(0, _definition_degree(d) - 1) for d in definitions.values())
    return (
        .45 * _nodes(node)
        + .75 * _depth(node)
        + .65 * sum(ops.values())
        + .75 * ops["call"]
        + 1.5 * inverse_defs
        + .35 * definition_degree
        + .7 * terminal_order
    )


def _max_bits(values):
    values = [_q(q) for q in values]
    return max(max(abs(q.numerator).bit_length(), q.denominator.bit_length()) for q in values)


def _digestible(q, cfg):
    q = _q(q)
    return (abs(q.numerator) <= cfg.max_answer_numerator and
            q.denominator <= cfg.max_answer_denominator)


def _visible_numbers(node, definitions):
    values = []

    def visit(cur):
        if cur["op"] == "poly":
            values.extend(_q(c) for c in cur["coeffs"])
        values.extend(_q(cur[k]) for k in ("shift",) if k in cur)
        for arg in cur.get("args", []):
            visit(arg)

    visit(node)
    for d in definitions.values():
        values.extend(_q(d[k]) for k in ("input_center", "output_center") if k in d)
        values.extend(_q(c) for c in d.get("coeffs", []))
    return set(values)


def _centered_expr(coeffs, center, variable):
    z = variable - sp.Rational(center.numerator, center.denominator)
    return sp.expand(sum(sp.Rational(c.numerator, c.denominator) * z**i
                         for i, c in enumerate(coeffs)))


def _exact_polynomial(node, definitions, root_center, variable=X):
    op = node["op"]
    if op == "var":
        return variable
    if op == "poly":
        return _centered_expr([_q(c) for c in node["coeffs"]], _q(node["center"]), variable)
    args = [_exact_polynomial(arg, definitions, root_center, variable) for arg in node["args"]]
    if op == "add":
        return sp.expand(args[0] + args[1])
    if op == "sub":
        return sp.expand(args[0] - args[1])
    if op == "mul":
        return sp.expand(args[0] * args[1])
    if op == "diff":
        return sp.diff(args[0], variable)
    if op == "integrate":
        t = sp.Dummy("t")
        body = _exact_polynomial(node["args"][0], definitions, root_center, t)
        return sp.integrate(body, (t, sp.Rational(root_center.numerator, root_center.denominator), variable))
    if op == "call":
        definition = definitions[node["name"]]
        if definition["kind"] != "explicit":
            raise ValueError("inverse function has no polynomial closed form")
        z = sp.Symbol("z")
        f = _centered_expr([_q(c) for c in definition["coeffs"]],
                           _q(definition["input_center"]), z)
        return sp.expand(f.subs(z, args[0]))
    raise ValueError(op)


def _format_polynomial(poly):
    poly = sp.Poly(poly, X, domain=sp.QQ)
    terms = []
    for (degree,), coeff in poly.terms():
        q = Fraction(int(coeff.p), int(coeff.q))
        if not q:
            continue
        sign = "-" if q < 0 else "+"
        q = abs(q)
        if degree == 0:
            body = _fmt(q)
        else:
            variable = "x" if degree == 1 else f"x^{degree}"
            body = variable if q == 1 else f"{_fmt(q)}*{variable}"
        terms.append((sign, body))
    if not terms:
        return "0"
    sign, body = terms[0]
    text = ("-" if sign == "-" else "") + body
    return text + "".join(f" {sign} {body}" for sign, body in terms[1:])


def _latex_q(q):
    return sp.latex(sp.Rational(q.numerator, q.denominator))



def _shifted(variable, center):
    center = _q(center)
    if not center:
        return variable
    sign = "-" if center > 0 else "+"
    return rf"\left({variable}{sign}{_latex_q(abs(center))}\right)"

def _render_poly(coeffs, center, variable):
    z = _shifted(variable, center)
    terms = []
    for degree, coeff in enumerate(coeffs):
        coeff = _q(coeff)
        if not coeff:
            continue
        sign = "-" if coeff < 0 else "+"
        coeff = abs(coeff)
        if degree == 0:
            body = _latex_q(coeff)
        else:
            power = z if degree == 1 else rf"{z}^{{{degree}}}"
            body = power if coeff == 1 else rf"{_latex_q(coeff)}{power}"
        terms.append((sign, body))
    if not terms:
        return "0"
    sign, body = terms[0]
    out = ("-" if sign == "-" else "") + body
    for sign, body in terms[1:]:
        out += f" {sign} {body}"
    return out


def _render_expr(node, root_center, variable="x", counter=None):
    counter = [0] if counter is None else counter
    op = node["op"]
    if op == "var":
        return variable
    if op == "poly":
        return _render_poly([_q(c) for c in node["coeffs"]], _q(node["center"]), variable)
    if op == "integrate":
        counter[0] += 1
        t = f"t_{{{counter[0]}}}"
        body = _render_expr(node["args"][0], root_center, t, counter)
        return rf"\int_{{{_latex_q(root_center)}}}^{{{variable}}}\left({body}\right)\,d{t}"

    args = [_render_expr(arg, root_center, variable, counter) for arg in node["args"]]
    if op == "add":
        return rf"\left({args[0]}\right)+\left({args[1]}\right)"
    if op == "sub":
        return rf"\left({args[0]}\right)-\left({args[1]}\right)"
    if op == "mul":
        return rf"\left({args[0]}\right)\left({args[1]}\right)"
    if op == "diff":
        return rf"\frac{{d}}{{d{variable}}}\left({args[0]}\right)"
    if op == "reciprocal":
        return rf"\frac{{1}}{{\left({args[0]}\right)}}"
    if op == "call":
        return rf"{node['name']}\left({args[0]}\right)"
    shift = _q(node["shift"])
    delta = args[0] if not shift else _shifted(rf"\left({args[0]}\right)", shift)
    if op == "exp0":
        return rf"\exp\left({delta}\right)"
    if op == "sin0":
        return rf"\sin\left({delta}\right)"
    if op == "log1p0":
        return rf"\log\left(1+{delta}\right)"
    raise ValueError(op)


def _render_definition(definition):
    if definition["kind"] == "explicit":
        coeffs = [_q(c) for c in definition["coeffs"]]
        center = _q(definition["input_center"])
        return rf"Let ${definition['name']}(x)={_render_poly(coeffs, center, 'x')}$."
    a = _q(definition["output_center"])
    b = _q(definition["input_center"])
    return (
        rf"Let ${definition['name']}$ be the unique local inverse of ${definition['base']}$ "
        rf"satisfying ${definition['name']}({_latex_q(b)})={_latex_q(a)}$; equivalently, "
        rf"${definition['base']}({definition['name']}(y))=y$ for $y$ near ${_latex_q(b)}$."
    )


def _safe_polynomial(text):
    text = str(text).strip().replace("^", "**")
    if not text or len(text) > 10_000:
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node, ast.Name) and node.id == "x":
            return X
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Integer and 0 <= int(right) <= 64:
                return left ** int(right)
        raise ValueError

    try:
        return sp.Poly(visit(tree.body), X, domain=sp.QQ)
    except (ValueError, TypeError, ZeroDivisionError, sp.PolynomialError):
        return None


def _parse_fraction(answer):
    try:
        return Fraction(str(answer).strip().replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


@dataclass
class FunctionManipulationConfig(Config):
    min_steps: int = 2
    max_steps: int = 3
    max_definitions: int = 2
    branch_degree: int = 1
    function_degree: int = 2
    coeff_bound: int = 2
    center_bound: int = 2
    max_terminal_order: int = 1
    max_internal_bits: int = 18
    max_answer_numerator: int = 12
    max_answer_denominator: int = 8
    max_expression_degree: int = 5
    max_expression_terms: int = 6
    min_reasoning_cost: float = 3.0
    max_reasoning_cost: float = 8.5
    expression_prob: float = .18
    definite_integral_prob: float = .10
    value_weight: float = .18
    calculus_weight: float = 1.0
    nonlinear_weight: float = .14
    composition_weight: float = .30
    inverse_weight: float = .08
    reuse_weight: float = .08
    identity_weight: float = .06
    trivial_answer_prob: float = .05
    literal_answer_prob: float = .08
    collapse_prob: float = .08
    min_terminal_step_fraction: float = .50
    n_trials: int = 700

    def apply_difficulty(self, level):
        self.min_steps = sround(self.min_steps + .55 * level)
        self.max_steps = sround(self.max_steps + 1.15 * level)
        self.max_definitions = sround(self.max_definitions + .45 * level)
        self.branch_degree = sround(self.branch_degree + .30 * level)
        self.function_degree = sround(self.function_degree + .35 * level)
        self.coeff_bound = sround(self.coeff_bound + .45 * level)
        self.center_bound = sround(self.center_bound + .25 * level)
        self.max_terminal_order = sround(self.max_terminal_order + .45 * level)
        self.max_internal_bits = sround(self.max_internal_bits + 3 * level)
        self.max_answer_numerator = sround(self.max_answer_numerator + 8 * level)
        self.max_answer_denominator = sround(self.max_answer_denominator + 4 * level)
        self.max_expression_degree = sround(self.max_expression_degree + level)
        self.max_expression_terms = sround(self.max_expression_terms + .8 * level)
        self.min_reasoning_cost += 1.0 * level
        self.max_reasoning_cost += 2.7 * level
        self.nonlinear_weight = min(.75, self.nonlinear_weight + .07 * level)
        self.composition_weight = min(1.0, self.composition_weight + .08 * level)
        self.inverse_weight = min(.65, self.inverse_weight + .07 * level)
        self.reuse_weight = min(.55, self.reuse_weight + .06 * level)
        self.identity_weight = min(.22, self.identity_weight + .02 * level)
        self.min_terminal_step_fraction = min(.70, self.min_terminal_step_fraction + .035 * level)


class FunctionManipulation(Task):
    config_cls = FunctionManipulationConfig
    summary = "Multistep symbolic function manipulation with composition, local inverses, calculus, and short exact answers."

    def generate_entry(self):
        cfg = self.config
        prec = cfg.max_terminal_order + cfg.max_steps + cfg.function_degree + 5

        for _ in range(cfg.n_trials):
            root_center = Fraction(random.randint(-cfg.center_bound, cfg.center_bound))
            current = {"op": "var"}
            definitions = {}
            name_source = _new_names()
            trace = [_eval(current, definitions, root_center, prec)]
            n_steps = random.randint(cfg.min_steps, cfg.max_steps)

            for _ in range(n_steps):
                for _ in range(60):
                    candidate, additions = _candidate(
                        current, trace[-1], definitions, name_source, root_center, cfg
                    )
                    if candidate is None:
                        continue
                    trial_defs = dict(definitions)
                    trial_defs.update({d["name"]: d for d in additions})
                    try:
                        value = _eval(candidate, trial_defs, root_center, prec)
                    except (ArithmeticError, ValueError, ZeroDivisionError):
                        continue
                    if _max_bits(value) > cfg.max_internal_bits:
                        continue
                    if value == trace[-1] and random.random() >= cfg.identity_weight:
                        continue
                    if not any(value[1:]) and random.random() >= cfg.collapse_prob:
                        continue
                    current, definitions = candidate, trial_defs
                    trace.append(value)
                    break
                else:
                    break
            else:
                ops = _ops(current)
                if len(ops) < cfg.min_steps or len(set(ops)) < 2:
                    continue
                if not ({"diff", "integrate", "call"} & set(ops)):
                    continue

                expression_mode = random.random() < cfg.expression_prob
                exact_poly = None
                if expression_mode and set(ops) <= POLY_OPS and all(
                    d["kind"] == "explicit" for d in definitions.values()
                ):
                    try:
                        exact_poly = sp.Poly(
                            _exact_polynomial(current, definitions, root_center), X, domain=sp.QQ
                        )
                        coeffs = [Fraction(int(c.p), int(c.q)) for c in exact_poly.all_coeffs()]
                        if (exact_poly.degree() > cfg.max_expression_degree or
                                len(exact_poly.terms()) > cfg.max_expression_terms or
                                not all(_digestible(c, cfg) for c in coeffs)):
                            exact_poly = None
                    except (ArithmeticError, ValueError, TypeError, sp.PolynomialError):
                        exact_poly = None

                if exact_poly is not None:
                    mode = "polynomial"
                    terminal_order = exact_poly.degree()
                    answer = _format_polynomial(exact_poly)
                else:
                    candidates = []
                    required_changes = max(1, math.ceil(cfg.min_terminal_step_fraction * n_steps))
                    visible = _visible_numbers(current, definitions)
                    for k in range(cfg.max_terminal_order + 1):
                        q = math.factorial(k) * trace[-1][k]
                        changes = sum(a[k] != b[k] for a, b in zip(trace, trace[1:]))
                        if changes < required_changes or not _digestible(q, cfg):
                            continue
                        if q in {0, 1, -1} and random.random() >= cfg.trivial_answer_prob:
                            continue
                        if q in visible and random.random() >= cfg.literal_answer_prob:
                            continue
                        weight = cfg.value_weight if k == 0 else 1 / k
                        candidates.append((k, q, weight, changes))
                    if not candidates:
                        continue
                    k, answer_value, _, changes = random.choices(
                        candidates, [item[2] for item in candidates]
                    )[0]
                    terminal_order = k
                    mode = "value" if k == 0 else "derivative"
                    answer = _fmt(answer_value)

                    if random.random() < cfg.definite_integral_prob and set(ops) <= POLY_OPS and all(
                        d["kind"] == "explicit" for d in definitions.values()
                    ):
                        try:
                            poly = _exact_polynomial(current, definitions, root_center)
                            upper = root_center + random.choice([-2, -1, 1, 2])
                            integral = sp.Rational(sp.integrate(poly, (X, root_center, upper)))
                            q = Fraction(int(integral.p), int(integral.q))
                            if _digestible(q, cfg) and q not in {0, 1, -1}:
                                mode, answer, terminal_order = "definite_integral", _fmt(q), 0
                            else:
                                upper = None
                        except (ArithmeticError, TypeError, ValueError):
                            upper = None
                    else:
                        upper = None

                cost = _reasoning_cost(current, definitions, terminal_order)
                if not cfg.min_reasoning_cost <= cost <= cfg.max_reasoning_cost:
                    continue

                trace_text = [
                    f"After operation {i}: value={_fmt(jet[0])}, derivative={_fmt(jet[1])}"
                    for i, jet in enumerate(trace[1:], 1)
                ]
                metadata = edict(
                    definitions=list(definitions.values()),
                    expression=current,
                    latex=_render_expr(current, root_center),
                    root_center=_fmt(root_center),
                    mode=mode,
                    order=terminal_order,
                    upper=_fmt(upper) if mode == "definite_integral" else None,
                    ops=ops,
                    depth=_depth(current),
                    reasoning_cost=cost,
                    cot="\n".join(trace_text),
                )
                return Entry(metadata=metadata, answer=answer)

        return None

    def render_prompt(self, metadata):
        lines = [_render_definition(d) for d in metadata.definitions]
        lines.append(rf"Define $h(x)={metadata.latex}$.")
        center = _latex_q(_q(metadata.root_center))
        if metadata.mode == "polynomial":
            lines.append("Simplify $h(x)$ to an expanded polynomial in $x$.")
            lines.append("The answer is an expanded polynomial with reduced rational coefficients.")
        elif metadata.mode == "value":
            lines.append(rf"Compute $h({center})$.")
            lines.append("The answer is a reduced rational number.")
        elif metadata.mode == "derivative":
            if metadata.order == 1:
                lines.append(rf"Compute $h'({center})$.")
            else:
                lines.append(rf"Compute $h^{{({metadata.order})}}({center})$.")
            lines.append("The answer is a reduced rational number.")
        else:
            upper = _latex_q(_q(metadata.upper))
            lines.append(rf"Compute $\int_{{{center}}}^{{{upper}}}h(x)\,dx$.")
            lines.append("The answer is a reduced rational number.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        if entry.metadata["mode"] == "polynomial":
            got, want = _safe_polynomial(answer), _safe_polynomial(entry.answer)
            return float(got == want) if got is not None and want is not None else 0.0
        got = _parse_fraction(answer)
        return float(got == Fraction(entry.answer)) if got is not None else 0.0

    def balancing_key(self, problem):
        ops = Counter(problem.metadata.ops)
        return str((
            problem.metadata.mode,
            int(any(d["kind"] == "inverse" for d in problem.metadata.definitions)),
            int("call" in ops),
            int("diff" in ops),
            int("integrate" in ops),
            int(any(op in ops for op in ADVANCED_OPS)),
            round(problem.metadata.reasoning_cost / 3),
        ))

    def deduplication_key(self, problem):
        return str((
            problem.metadata.definitions,
            problem.metadata.expression,
            problem.metadata.mode,
            problem.metadata.order,
            problem.metadata.upper,
        ))
