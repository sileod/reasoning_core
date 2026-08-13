"""Procedural Lean 4 theorem generation, Mathlib-backed.

Lean itself is the oracle. Install (elan + Mathlib + REPL) lives under appdirs,
so nothing leaks into the user's home or the project tree.
"""

import ast
from difflib import SequenceMatcher
import json
import os
import platform
import queue
import random
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from urllib.request import urlretrieve

import networkx as nx
from appdirs import AppDirs
from easydict import EasyDict as edict
from gramforge import generate, init_grammar
from reasoning_core.template import Config, Entry, Task, DevTask


# ============================================================================
# Appdir-managed Lean + Mathlib install
# ============================================================================

_DIRS = AppDirs("reasoning_core")
_BASE = Path(_DIRS.user_data_dir) / "lean"
_ELAN_HOME = _BASE / "elan"
_PROJECT_MATHLIB = _BASE / "project"
_PROJECT_CORE = _BASE / "project_core"
_PROJECT = _PROJECT_MATHLIB
_LEAN_TOOLCHAIN = "leanprover/lean4:v4.29.0"
_MATHLIB_REV = "v4.29.0"
_REPL_REV = "v4.29.0"
_ELAN_RELEASE = "v4.1.2"
_PRELUDE_IMPORTS_MATHLIB = "import Mathlib"
_PRELUDE_IMPORTS_CORE = "import Std"
_PRELUDE_IMPORTS = _PRELUDE_IMPORTS_MATHLIB
_LEAN_IMPORT_CMD = "import ReasoningCorePrelude"
_ALLOW_SOURCE_BUILD_ENV = "REASONING_CORE_LEAN_ALLOW_SOURCE_BUILD"

BANNED_LEAN_TOKENS = (
    "sorry", "admit", "unsafe", "#eval", "native_decide",
    "set_option", "opaque", "syntax ", "macro ",
)

_LAKEFILE = f"""\
import Lake
open Lake DSL

package leanrepl

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "{_MATHLIB_REV}"

require repl from git
  "https://github.com/leanprover-community/repl" @ "{_REPL_REV}"

lean_lib ReasoningCorePrelude
"""

_LAKEFILE_CORE = f"""\
import Lake
open Lake DSL

package leanrepl_core

require repl from git
  "https://github.com/leanprover-community/repl" @ "{_REPL_REV}"

lean_lib ReasoningCorePrelude
"""


def _project(use_mathlib=True):
    return _PROJECT_MATHLIB if use_mathlib else _PROJECT_CORE


def _lakefile(use_mathlib=True):
    return _LAKEFILE if use_mathlib else _LAKEFILE_CORE


def _prelude_imports(use_mathlib=True):
    return _PRELUDE_IMPORTS_MATHLIB if use_mathlib else _PRELUDE_IMPORTS_CORE


def _install_stamp(use_mathlib=True):
    return _project(use_mathlib) / ".reasoning_core_lean_env.json"


def _profile_stamp(use_mathlib=True):
    stamp = {
        "lean": _LEAN_TOOLCHAIN,
        "repl": _REPL_REV,
        "prelude": _prelude_imports(use_mathlib),
        "profile": "mathlib" if use_mathlib else "core",
    }
    if use_mathlib:
        stamp["mathlib"] = _MATHLIB_REV
    return stamp


def _profile_ready(use_mathlib=True):
    project = _project(use_mathlib)
    repl_bin = project / ".lake" / "packages" / "repl" / ".lake" / "build" / "bin" / "repl"
    prelude_olean = project / ".lake" / "build" / "lib" / "lean" / "ReasoningCorePrelude.olean"
    ready = repl_bin.exists() and prelude_olean.exists()
    if use_mathlib:
        mathlib_olean = project / ".lake" / "packages" / "mathlib" / ".lake" / "build" / "lib" / "lean" / "Mathlib.olean"
        ready = ready and mathlib_olean.exists()
    try:
        return ready and json.loads(_install_stamp(use_mathlib).read_text()) == _profile_stamp(use_mathlib)
    except Exception:
        return False


def _elan_env():
    env = os.environ.copy()
    env["ELAN_HOME"] = str(_ELAN_HOME)
    env["PATH"] = f"{_ELAN_HOME / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def _elan_bin(name):
    suffix = ".exe" if platform.system().lower() == "windows" else ""
    return _ELAN_HOME / "bin" / f"{name}{suffix}"


def _install_elan():
    if not _elan_bin("lean").exists():
        _BASE.mkdir(parents=True, exist_ok=True)
        sys_, mach = platform.system().lower(), platform.machine().lower()
        arch = "aarch64" if ("arm" in mach or "aarch64" in mach) else "x86_64"
        tgts = {"linux": f"{arch}-unknown-linux-gnu", "darwin": f"{arch}-apple-darwin"}
        if sys_ not in tgts:
            raise RuntimeError(f"Unsupported platform for auto-install: {sys_}")
        url = (f"https://github.com/leanprover/elan/releases/download/"
               f"{_ELAN_RELEASE}/elan-{tgts[sys_]}.tar.gz")
        with tempfile.TemporaryDirectory() as td:
            tar = Path(td) / "elan.tar.gz"
            urlretrieve(url, tar)
            subprocess.run(["tar", "xzf", str(tar), "-C", td], check=True)
            subprocess.run(
                [str(Path(td) / "elan-init"), "-y", "--no-modify-path",
                 "--default-toolchain", _LEAN_TOOLCHAIN],
                env=_elan_env(), check=True,
            )
    installed = subprocess.run(
        [str(_elan_bin("elan")), "toolchain", "list"],
        env=_elan_env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    ).stdout
    if _LEAN_TOOLCHAIN not in installed:
        subprocess.run([str(_elan_bin("elan")), "toolchain", "install", _LEAN_TOOLCHAIN],
                       env=_elan_env(), check=True)
    subprocess.run([str(_elan_bin("elan")), "default", _LEAN_TOOLCHAIN],
                   env=_elan_env(), check=True)


def _ensure_project(use_mathlib=True):
    """Build a Lake project with the requested Lean profile. Returns repl binary."""
    project = _project(use_mathlib)
    prelude_imports = _prelude_imports(use_mathlib)
    repl_bin = project / ".lake" / "packages" / "repl" / ".lake" / "build" / "bin" / "repl"
    prelude = project / "ReasoningCorePrelude.lean"
    prelude_olean = project / ".lake" / "build" / "lib" / "lean" / "ReasoningCorePrelude.olean"
    mathlib_olean = project / ".lake" / "packages" / "mathlib" / ".lake" / "build" / "lib" / "lean" / "Mathlib.olean"
    stamp = _profile_stamp(use_mathlib)
    build_ready = repl_bin.exists() and prelude_olean.exists()
    if use_mathlib:
        build_ready = build_ready and mathlib_olean.exists()
    install_stamp = _install_stamp(use_mathlib)
    if build_ready and install_stamp.exists():
        try:
            if json.loads(install_stamp.read_text()) == stamp:
                return repl_bin
        except Exception:
            pass
    if build_ready and prelude.exists():
        if prelude.read_text() == prelude_imports + "\n":
            install_stamp.write_text(json.dumps(stamp, sort_keys=True))
            return repl_bin
    _install_elan()
    project.mkdir(parents=True, exist_ok=True)
    (project / "lakefile.lean").write_text(_lakefile(use_mathlib))
    (project / "lean-toolchain").write_text(_LEAN_TOOLCHAIN + "\n")
    (project / "ReasoningCorePrelude.lean").write_text(prelude_imports + "\n")
    env = _elan_env()
    subprocess.run(["lake", "update"], cwd=project, env=env, check=True)
    if use_mathlib:
        # mathlib cache: skip the multi-hour olean compile. If cache download fails,
        # fail fast unless the caller explicitly opted into a source build.
        cache = subprocess.run(["lake", "exe", "cache", "get"], cwd=project, env=env)
        if cache.returncode != 0 or not mathlib_olean.exists():
            if env.get(_ALLOW_SOURCE_BUILD_ENV) != "1":
                raise RuntimeError(
                    "Lean Mathlib binary cache was not available, so building would "
                    "compile Mathlib from source and can take hours. Fix the cache "
                    "download, or set "
                    f"{_ALLOW_SOURCE_BUILD_ENV}=1 to allow the slow source build."
                )
    subprocess.run(["lake", "build", "ReasoningCorePrelude"], cwd=project, env=env, check=True)
    subprocess.run(["lake", "build", "repl"], cwd=project, env=env, check=True)
    if not repl_bin.exists():
        raise RuntimeError(f"REPL binary missing after build: {repl_bin}")
    install_stamp.write_text(json.dumps(stamp, sort_keys=True))
    return repl_bin


def ensure_lean_mathlib(verbose=True):
    """Install or reuse Lean, Mathlib, and leanprover-community/repl in appdirs."""
    repl_bin = _ensure_project(use_mathlib=True)
    if verbose:
        print(f"Lean toolchain: {_LEAN_TOOLCHAIN}")
        print(f"Mathlib project: {_PROJECT_MATHLIB}")
        print(f"REPL binary: {repl_bin}")
    return repl_bin


def ensure_lean_core(verbose=True):
    """Install or reuse Lean, Std, and leanprover-community/repl in appdirs."""
    repl_bin = _ensure_project(use_mathlib=False)
    if verbose:
        print(f"Lean toolchain: {_LEAN_TOOLCHAIN}")
        print(f"Lean core project: {_PROJECT_CORE}")
        print(f"REPL binary: {repl_bin}")
    return repl_bin


# ============================================================================
# REPL wrapper — Mathlib loaded into a persistent environment handle
# ============================================================================

class LeanRunner:
    def __init__(self, timeout=360, use_mathlib=True):
        self.timeout = timeout
        self.use_mathlib = use_mathlib
        self.cache = {}
        self.proc = None
        self.stdout = queue.Queue()
        self._mathlib_env = None
        self._start()

    def _start(self):
        self.close()
        project = _project(self.use_mathlib)
        repl_bin = _ensure_project(use_mathlib=self.use_mathlib)
        self.proc = subprocess.Popen(
            ["lake", "env", str(repl_bin)],
            cwd=project, env=_elan_env(),
            text=True, encoding="utf-8",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        threading.Thread(target=self._read, daemon=True).start()
        resp = self._send_raw({"cmd": _LEAN_IMPORT_CMD})
        msgs = resp.get("messages") or []
        errors = [m for m in msgs if m.get("severity") == "error"]
        self._mathlib_env = resp.get("env")
        if errors or self._mathlib_env is None:
            diag = "\n".join(str(m.get("data", m)) for m in errors or msgs)
            self.close()
            raise RuntimeError(f"Lean REPL failed to import prelude: {diag}")

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self.stdout.put(line)

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def _send_raw(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n\n")
        self.proc.stdin.flush()
        chunks = []
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Lean REPL timeout")
            try:
                chunks.append(self.stdout.get(timeout=remaining))
            except queue.Empty:
                raise TimeoutError("Lean REPL timeout")
            try:
                return json.loads("\n".join(chunks))
            except json.JSONDecodeError:
                continue

    def check(self, code):
        if code in self.cache:
            return self.cache[code]
        payload = {"cmd": code}
        if self._mathlib_env is not None:
            payload["env"] = self._mathlib_env
        try:
            result = self._send_raw(payload)
        except Exception as e:
            self._start()
            return False, str(e)
        msgs = result.get("messages") or []
        has_error = any(m.get("severity") == "error" for m in msgs)
        has_sorry = bool(result.get("sorries"))
        diag = "\n".join(str(m.get("data", m)) for m in msgs)
        out = (not has_error and not has_sorry, diag)
        self.cache[code] = out
        return out


_GLOBAL_RUNNERS = {}
def get_runner(use_mathlib=True):
    if use_mathlib not in _GLOBAL_RUNNERS:
        _GLOBAL_RUNNERS[use_mathlib] = LeanRunner(use_mathlib=use_mathlib)
    return _GLOBAL_RUNNERS[use_mathlib]


def _mget(metadata, key):
    return metadata[key] if isinstance(metadata, dict) else getattr(metadata, key)


def _safe(text):
    low = str(text).lower()
    return not any(tok in low for tok in BANNED_LEAN_TOKENS)


# ============================================================================
# Config + gramforge grammars
# ============================================================================

@dataclass
class LeanConfig(Config):
    n_vars: int = 2
    expr_depth: int = 3
    n_hyps: int = 2
    n_candidates: int = 6
    multiple_choice_prob: float = 0.2
    use_mathlib: bool = True

    def apply_difficulty(self, level):
        self.n_vars += level
        self.expr_depth += level
        self.n_hyps += 2 * level + max(0, level - 2)
        self.n_candidates += level

@dataclass
class LeanDerivationNode:
    clause_id: str
    clause_formula: str
    parents: tuple[str, ...] = ()
    inference: str = ""
    role: str = "plain"
    interesting_score: float = 0.0
    full_cnf_clause: str = ""
    proof: str = ""
    depth: int = 0


@dataclass(frozen=True)
class LeanSchema:
    kind: str
    decl: str
    hyps: tuple[str, ...]
    goal: str
    proof: str
    slots: tuple[str, ...] = ()
    slot_type: str = "Int"
    tactic_fallbacks: tuple[str, ...] = ("aesop", "simp_all", "omega", "tauto")


VAR_NAMES = tuple("abcdefghijkmn")
def _vars(n):
    return list(VAR_NAMES[: max(1, n)])


def _int_lin_grammar(vars_, max_coef=4):
    """Linear Int expressions: sums/differences of variables with small coefficients."""
    g = init_grammar(["lean"], preprocess_template=lambda s: s)
    g("start(e)", "{0}")
    g("e(atom)", "{0}", weight=3)
    g("e(e,e)", "({0} + {1})")
    g("e(e,e)", "({0} - {1})")
    g("atom", "0")
    g("atom", "1")
    for c in range(2, max_coef + 1):
        g("atom", str(c))
    for v in vars_:
        g("atom", v)
        for c in range(2, max_coef + 1):
            g("atom", f"{c} * {v}")
    return g


def _int_poly_grammar(vars_, max_coef=3):
    g = _int_lin_grammar(vars_, max_coef)
    g("e(e,e)", "({0} * {1})", weight=2)
    for v in vars_:
        g("atom", f"{v}^2")
    return g


def _prop_grammar(atoms):
    g = init_grammar(["lean"], preprocess_template=lambda s: s)
    g("start(p)", "{0}")
    g("p(atom)", "{0}", weight=3)
    g("p(p,p)", "({0} ∧ {1})")
    g("p(p,p)", "({0} ∨ {1})")
    g("p(p,p)", "({0} → {1})")
    g("p(p)", "(¬ {0})")
    for a in atoms:
        g("atom", a)
    return g


def _sample(grammar, depth):
    return generate(grammar, depth=depth, mode="sequential") @ "lean"


def _used_int_vars(*texts):
    return sorted(set(re.findall(r"\b([a-n])\b", " ".join(map(str, texts)))))


# ============================================================================
# Strategy 1: gramforge expressions + Lean tactic oracle
# ============================================================================

# Ring axioms applied as string rewrites: produce a syntactically distinct but
# provably-equal form. `ring` will close the resulting equality.
_RING_REWRITES = (
    (re.compile(r"\(([^()]+) \+ ([^()]+)\)"),
     lambda m: f"({m.group(2)} + {m.group(1)})"),
    (re.compile(r"\(([^()]+) \* ([^()]+)\)"),
     lambda m: f"({m.group(2)} * {m.group(1)})"),
    (re.compile(r"\(([^()]+) \* \(([^()]+) \+ ([^()]+)\)\)"),
     lambda m: f"({m.group(1)} * {m.group(2)} + {m.group(1)} * {m.group(3)})"),
)


def _ring_equiv(expr, names):
    s = expr
    for _ in range(random.randint(1, 4)):
        rule, repl = random.choice(_RING_REWRITES)
        m = rule.search(s)
        if m:
            s = s[: m.start()] + repl(m) + s[m.end():]
    if random.random() < 0.4 and names:
        s = f"({s} + 0 * {random.choice(names)})"
    return s


def gen_polynomial_eq(config):
    names = _vars(int(config.n_vars))
    g = _int_poly_grammar(names)
    lhs = _sample(g, int(config.expr_depth))
    rhs = _ring_equiv(lhs, names)
    if rhs == lhs:
        return None
    return edict(
        decl=f"({' '.join(names)} : Int)", hyps=[],
        goal=f"{lhs} = {rhs}", primary="ring", kind="poly_eq",
    )


def gen_linear_ineq(config):
    """Bound each variable, then ask omega about a positive linear combination of bounds."""
    n = max(2, int(config.n_vars))
    names = _vars(n)
    hyps, bounds = [], {}
    for i, v in enumerate(names):
        b = random.randint(0, 20)
        op = random.choice(["≤", "<"])
        bounds[v] = (b, op)
        hyps.append((f"h{i}", f"{v} {op} {b}"))
    coefs = {v: random.randint(1, 4) for v in names}
    lhs = " + ".join(f"{c} * {v}" if c > 1 else v for v, c in coefs.items())
    upper = sum(c * bounds[v][0] for v, c in coefs.items())
    upper += random.randint(0, 5)  # slack keeps it provable
    op = "≤" if all(o == "≤" for _, o in bounds.values()) else "<"
    return edict(
        decl=f"({' '.join(names)} : Int)", hyps=hyps,
        goal=f"{lhs} {op} {upper}",
        primary="omega", kind="lin_ineq",
    )


_ORDER_RULES = {
    ("≤", "≤"): ("Int.le_trans", "≤"),
    ("<", "<"): ("Int.lt_trans", "<"),
    ("<", "≤"): ("Int.lt_of_lt_of_le", "<"),
    ("≤", "<"): ("Int.lt_of_le_of_lt", "<"),
}


def gen_order_chain(config):
    """Random mixed strict/nonstrict chains with an explicit transitivity proof."""
    n = max(2, int(config.n_hyps))
    names = _vars(max(2, int(config.n_vars)))
    g = _int_lin_grammar(names, max_coef=3)
    terms = []
    for _ in range(n + 1):
        for _ in range(12):
            t = _sample(g, max(1, int(config.expr_depth) - 1))
            if not re.search(r"[a-n]", t):
                continue  # reject pure-constant terms (avoid vacuous hyps like 3 ≤ 2)
            if t not in terms:  # all-distinct avoids hyp cycles like a ≤ 3b ≤ a
                terms.append(t)
                break
        else:
            return None
    if len(set(terms)) < 2:
        return None
    strict_at = random.randrange(n)
    hyps, ops = [], []
    for i, (a, b) in enumerate(zip(terms, terms[1:])):
        op = "<" if i == strict_at and random.random() < 0.5 else "≤"
        hyps.append((f"h{i}", f"{a} {op} {b}"))
        ops.append(op)
    proof, current = "h0", ops[0]
    for i, op in enumerate(ops[1:], 1):
        theorem, current = _ORDER_RULES[(current, op)]
        proof = f"{theorem} ({proof}) h{i}"
    used = _used_int_vars(*terms)
    return edict(
        decl=f"({' '.join(used)} : Int)" if used else "",
        hyps=hyps, goal=f"{terms[0]} {current} {terms[-1]}",
        primary=f"exact {proof}", kind="order_chain",
    )


def gen_divisibility(config):
    names = _vars(max(2, int(config.n_vars)))
    a, b = names[0], names[1]
    k = random.randint(2, 6)
    c1, c2 = random.randint(1, 4), random.randint(1, 4)
    sign = random.choice(["+", "-"])
    expr = f"{c1} * {a} {sign} {c2} * {b}"
    combine = "dvd_add" if sign == "+" else "dvd_sub"
    proof = (
        f"exact {combine} (dvd_mul_of_dvd_right h1 {c1}) "
        f"(dvd_mul_of_dvd_right h2 {c2})"
    )
    return edict(
        decl=f"({' '.join(names)} : Int)",
        hyps=[("h1", f"({k} : Int) ∣ {a}"), ("h2", f"({k} : Int) ∣ {b}")],
        goal=f"({k} : Int) ∣ ({expr})", primary=proof, kind="div",
    )


def gen_concrete(config):
    a = random.randint(1, 100)
    b = random.randint(1, 50)
    op = random.choice(["+", "*", "%"])
    res = {"+": a + b, "*": a * b, "%": a % b}[op]
    return edict(
        decl="", hyps=[],
        goal=f"({a} {op} {b} : Nat) = {res}", primary="decide", kind="concrete",
    )


_TAUT_TEMPLATES = (
    "({p} ∧ {q}) → {p}",
    "({p} ∧ {q}) → {q}",
    "{p} → ({p} ∨ {q})",
    "({p} → {q}) → (¬ {q} → ¬ {p})",
    "({p} ∨ {q}) ∧ ¬ {p} → {q}",
    "({p} → {q}) ∧ ({q} → {r}) → ({p} → {r})",
    "¬ ({p} ∧ ¬ {p})",
    "{p} ∨ ¬ {p}",
    "({p} → {q}) → (({p} ∧ {r}) → ({q} ∧ {r}))",
    "(({p} ∧ {q}) ∨ ({p} ∧ {r})) → ({p} ∧ ({q} ∨ {r}))",
)


def gen_tautology(config):
    """Tautology schema instantiated with disjoint propositional subformulas.

    Each slot in the template is bound to a subformula over a *disjoint* atom
    set, so the tautology doesn't collapse to a one-atom triviality.
    """
    tmpl = random.choice(_TAUT_TEMPLATES)
    slots = sorted(set(re.findall(r"\{(\w+)\}", tmpl)))
    atoms_pool = [f"p{i}" for i in range(2 * len(slots) + 1)]
    random.shuffle(atoms_pool)
    bindings = {}
    for s in slots:
        # 1–2 atoms per slot, drawn without replacement
        k = 1 if len(atoms_pool) < 2 or random.random() < 0.4 else 2
        chunk, atoms_pool = atoms_pool[:k], atoms_pool[k:]
        g = _prop_grammar(chunk)
        bindings[s] = _sample(g, max(1, int(config.expr_depth) - 1))
    formula = tmpl.format(**bindings)
    used = sorted(set(re.findall(r"\bp\d+\b", formula)))
    decl = f"({' '.join(used)} : Prop)" if used else "(p0 : Prop)"
    return edict(decl=decl, hyps=[], goal=formula, primary="tauto", kind="taut")


def gen_finset_identity(config):
    names = list("stuvwxyz")[: max(2, min(6, int(config.n_vars) + 1))]
    a, b, c = random.sample(names, 3)
    templates = (
        (f"{a} ∩ ({b} ∪ {c}) = ({a} ∩ {b}) ∪ ({a} ∩ {c})", f"simpa using inf_sup_left {a} {b} {c}"),
        (f"({a} ∪ {b}) ∩ {c} = ({a} ∩ {c}) ∪ ({b} ∩ {c})", f"simpa using inf_sup_right {a} {b} {c}"),
        (f"{a} ∪ ({b} ∩ {c}) = ({a} ∪ {b}) ∩ ({a} ∪ {c})", f"simpa using sup_inf_left {a} {b} {c}"),
        (f"{a} ∩ ({b} ∩ {c}) = ({a} ∩ {b}) ∩ {c}", f"simpa using inf_assoc {a} {b} {c}"),
        (f"{a} ∪ ({b} ∪ {c}) = ({a} ∪ {b}) ∪ {c}", f"simpa using sup_assoc {a} {b} {c}"),
    )
    goal, proof = random.choice(templates)
    return edict(
        decl=f"({' '.join(names)} : Finset Nat)", hyps=[],
        goal=goal, primary=proof, kind="finset_identity",
    )


def gen_set_monotone(config):
    names = list("stuvwxyz")[: max(3, min(6, int(config.n_vars) + 1))]
    s, t, u = random.sample(names, 3)
    cases = (
        ([(f"h0", f"{s} ⊆ {t}")], f"{s} ∩ {u} ⊆ {t} ∩ {u}", "exact Set.inter_subset_inter h0 subset_rfl"),
        ([(f"h0", f"{s} ⊆ {t}")], f"{u} ∩ {s} ⊆ {u} ∩ {t}", "exact Set.inter_subset_inter subset_rfl h0"),
        ([(f"h0", f"{s} ⊆ {t}")], f"{s} ∪ {u} ⊆ {t} ∪ {u}", "exact Set.union_subset_union h0 subset_rfl"),
        ([(f"h0", f"{s} ⊆ {t}")], f"{u} ∪ {s} ⊆ {u} ∪ {t}", "exact Set.union_subset_union subset_rfl h0"),
    )
    hyps, goal, proof = random.choice(cases)
    return edict(
        decl=f"({' '.join(names)} : Set Int)", hyps=hyps,
        goal=goal, primary=proof, kind="set_monotone",
    )


def gen_list_length(config):
    names = list("xyzuvw")[: max(2, min(5, int(config.n_vars)))]
    xs = random.sample(names, random.randint(2, min(4, len(names))))
    concat = " ++ ".join(xs)
    length_sum = " + ".join(f"{x}.length" for x in xs)
    return edict(
        decl=f"({' '.join(xs)} : List Nat)", hyps=[],
        goal=f"({concat}).length = {length_sum}",
        primary="simp [List.length_append, Nat.add_assoc]",
        kind="list_length",
    )


def gen_core_prop_chain(config):
    n_hyps = max(2, min(6, int(config.n_hyps)))
    props = [f"p{i}" for i in range(n_hyps + 1)]
    hyps = [(f"h{i}", f"{props[i]} → {props[i + 1]}") for i in range(n_hyps)]
    expr = "hp"
    for i in range(n_hyps):
        expr = f"h{i} ({expr})"
    proof = f"fun hp => {expr}"
    return edict(
        decl=f"({' '.join(props)} : Prop)",
        hyps=hyps,
        goal=f"{props[0]} → {props[-1]}",
        primary=f"exact {proof}",
        kind="core_prop_chain",
    )


def gen_core_and(config):
    cases = (
        ("p ∧ q → p", "intro h; exact h.1"),
        ("p ∧ q → q", "intro h; exact h.2"),
        ("p → q → p ∧ q", "intro hp hq; exact And.intro hp hq"),
        ("p ∧ q → q ∧ p", "intro h; exact And.intro h.2 h.1"),
        ("(p ∧ q) ∧ r → p ∧ (q ∧ r)",
         "intro h; exact And.intro h.1.1 (And.intro h.1.2 h.2)"),
    )
    goal, proof = random.choice(cases)
    return edict(decl="(p q r : Prop)", hyps=[], goal=goal,
                 primary=proof, kind="core_and")


def gen_core_or(config):
    cases = (
        ("p → p ∨ q", "intro hp; exact Or.inl hp"),
        ("q → p ∨ q", "intro hq; exact Or.inr hq"),
        ("p ∨ q → q ∨ p",
         "intro h; cases h with | inl hp => exact Or.inr hp | inr hq => exact Or.inl hq"),
        ("(p → r) → (q → r) → p ∨ q → r",
         "intro hp hq h; cases h with | inl h0 => exact hp h0 | inr h1 => exact hq h1"),
    )
    goal, proof = random.choice(cases)
    return edict(decl="(p q r : Prop)", hyps=[], goal=goal,
                 primary=proof, kind="core_or")


def gen_core_nat_decide(config):
    a = random.randint(0, 30)
    b = random.randint(0, 30)
    op = random.choice(["+", "*"])
    res = {"+": a + b, "*": a * b}[op]
    return edict(
        decl="", hyps=[],
        goal=f"({a} {op} {b} : Nat) = {res}",
        primary="decide", kind="core_nat_decide",
    )


def gen_core_list(config):
    names = list("xyzuvw")[: max(2, min(4, int(config.n_vars)))]
    xs, ys = random.sample(names, 2)
    cases = (
        (f"([] ++ {xs} : List Nat) = {xs}", "rfl"),
        (f"({xs} ++ [] : List Nat) = {xs}", "simp"),
        (f"List.length ({xs} ++ {ys}) = List.length {xs} + List.length {ys}",
         "simp [List.length_append]"),
    )
    goal, proof = random.choice(cases)
    return edict(decl=f"({' '.join(names)} : List Nat)", hyps=[],
                 goal=goal, primary=proof, kind="core_list")


# ============================================================================
# Strategy 2: Mathlib lemma schemas + gramforge slot instantiation
# ============================================================================

_LEMMA_SCHEMAS = (
    LeanSchema(
        kind="le_trans",
        decl="(α : Type*) [Preorder α] (a b c : α)",
        hyps=("a ≤ b", "b ≤ c"),
        goal="a ≤ c",
        proof="exact le_trans h0 h1",
    ),
    LeanSchema(
        kind="lt_trans",
        decl="(α : Type*) [Preorder α] (a b c : α)",
        hyps=("a < b", "b < c"),
        goal="a < c",
        proof="exact lt_trans h0 h1",
    ),
    LeanSchema(
        kind="lt_of_lt_of_le",
        decl="(α : Type*) [Preorder α] (a b c : α)",
        hyps=("a < b", "b ≤ c"),
        goal="a < c",
        proof="exact lt_of_lt_of_le h0 h1",
    ),
    LeanSchema(
        kind="lt_of_le_of_lt",
        decl="(α : Type*) [Preorder α] (a b c : α)",
        hyps=("a ≤ b", "b < c"),
        goal="a < c",
        proof="exact lt_of_le_of_lt h0 h1",
    ),
    LeanSchema(
        kind="add_le_add",
        decl="",
        hyps=("{a} ≤ {b}", "{c} ≤ {d}"),
        goal="{a} + {c} ≤ {b} + {d}",
        proof="exact add_le_add h0 h1",
        slots=("a", "b", "c", "d"),
    ),
    LeanSchema(
        kind="add_lt_add",
        decl="",
        hyps=("{a} < {b}", "{c} < {d}"),
        goal="{a} + {c} < {b} + {d}",
        proof="exact add_lt_add h0 h1",
        slots=("a", "b", "c", "d"),
    ),
    LeanSchema(
        kind="add_le_add_left",
        decl="",
        hyps=("{a} ≤ {b}",),
        goal="{c} + {a} ≤ {c} + {b}",
        proof="exact add_le_add_right h0 {c}",
        slots=("a", "b", "c"),
    ),
    LeanSchema(
        kind="add_le_add_right",
        decl="",
        hyps=("{a} ≤ {b}",),
        goal="{a} + {c} ≤ {b} + {c}",
        proof="exact add_le_add_left h0 {c}",
        slots=("a", "b", "c"),
    ),
    LeanSchema(
        kind="dvd_add",
        decl="",
        hyps=("{k} ∣ {a}", "{k} ∣ {b}"),
        goal="{k} ∣ ({a} + {b})",
        proof="exact dvd_add h0 h1",
        slots=("k", "a", "b"),
    ),
    LeanSchema(
        kind="dvd_sub",
        decl="",
        hyps=("{k} ∣ {a}", "{k} ∣ {b}"),
        goal="{k} ∣ ({a} - {b})",
        proof="exact dvd_sub h0 h1",
        slots=("k", "a", "b"),
    ),
    LeanSchema(
        kind="dvd_mul_of_dvd_left",
        decl="",
        hyps=("{a} ∣ {b}",),
        goal="{a} ∣ ({b} * {c})",
        proof="exact dvd_mul_of_dvd_left h0 {c}",
        slots=("a", "b", "c"),
    ),
    LeanSchema(
        kind="dvd_mul_of_dvd_right",
        decl="",
        hyps=("{a} ∣ {b}",),
        goal="{a} ∣ ({c} * {b})",
        proof="exact dvd_mul_of_dvd_right h0 {c}",
        slots=("a", "b", "c"),
    ),
    LeanSchema(
        kind="dvd_trans",
        decl="",
        hyps=("{a} ∣ {b}", "{b} ∣ {c}"),
        goal="{a} ∣ {c}",
        proof="exact dvd_trans h0 h1",
        slots=("a", "b", "c"),
    ),
    LeanSchema(
        kind="abs_nonneg",
        decl="",
        hyps=(),
        goal="0 ≤ |{a}|",
        proof="exact abs_nonneg {a}",
        slots=("a",),
    ),
    LeanSchema(
        kind="abs_add_le",
        decl="",
        hyps=(),
        goal="|{a} + {b}| ≤ |{a}| + |{b}|",
        proof="exact abs_add_le {a} {b}",
        slots=("a", "b"),
    ),
    LeanSchema(
        kind="sq_nonneg",
        decl="",
        hyps=(),
        goal="0 ≤ {a} ^ 2",
        proof="exact sq_nonneg {a}",
        slots=("a",),
    ),
    LeanSchema(
        kind="mul_self_nonneg",
        decl="",
        hyps=(),
        goal="0 ≤ {a} * {a}",
        proof="exact mul_self_nonneg {a}",
        slots=("a",),
    ),
)


def gen_lemma(config):
    schema = random.choice(_LEMMA_SCHEMAS)
    bindings = {}
    if schema.slots:
        g = _int_lin_grammar(_vars(int(config.n_vars)), max_coef=3)
        bindings = {
            s: _sample(g, max(1, int(config.expr_depth) - 1))
            for s in schema.slots
        }
    proof_bindings = {
        k: f"({v} : {schema.slot_type})" if schema.slots else f"({v})"
        for k, v in bindings.items()
    }
    expr_bindings = proof_bindings if schema.slots else bindings
    hyps = [(f"h{i}", t.format(**expr_bindings)) for i, t in enumerate(schema.hyps)]
    goal = schema.goal.format(**expr_bindings)
    decl = schema.decl
    if schema.slots:
        used = _used_int_vars(*bindings.values())
        decl = f"({' '.join(used)} : {schema.slot_type})" if used else ""
    proof = schema.proof.format(**proof_bindings)
    return edict(
        decl=decl, hyps=hyps, goal=goal, primary=proof,
        tactic_fallbacks=schema.tactic_fallbacks,
        kind=f"lemma:{schema.kind}",
    )


STRATEGIES = (
    gen_polynomial_eq, gen_linear_ineq, gen_order_chain,
    gen_divisibility, gen_tautology,
    gen_finset_identity, gen_set_monotone, gen_list_length,
    gen_lemma,
)

CORE_STRATEGIES = (
    gen_core_prop_chain, gen_core_and, gen_core_or,
    gen_core_nat_decide, gen_core_list,
)


# ============================================================================
# Instance assembly
# ============================================================================

def _render(inst, name="ex"):
    hyp_str = " ".join(f"({h} : {b})" for h, b in inst.hyps)
    args = " ".join(x for x in (inst.decl, hyp_str) if x)
    args = f" {args}" if args else ""
    return f"theorem {name}{args} : {inst.goal} := by\n"


def _is_theorem_specific_candidate(candidate):
    """Heuristic filter for candidates that depend on local theorem structure."""
    s = str(candidate)
    if re.search(r"\bh\d+\b", s):
        return True
    if re.search(r"\bhx\b|\bhp\b|\bhq\b|\bexact\s+[A-Za-z_][\w.]*\s+h", s):
        return True
    if re.match(r"(?:exact|simpa using)\s+[A-Za-z_][\w.']*(?:\s+\S+)+$", s):
        return True
    if any(tok in s for tok in ("le_trans", "lt_trans", "dvd_", "And.intro", "Or.inl", "Or.inr", "rcases", "cases ")):
        return True
    return False


def _hard_candidate_pairs(inst):
    """Recover the already Lean-checked corruption pairs from an instance."""
    if not _is_theorem_specific_candidate(inst.primary):
        return []
    negatives = [
        c for c, ok in zip(inst.candidates, inst.labels)
        if not ok and _is_theorem_specific_candidate(c)
    ]
    return [(inst.primary, negative) for negative in negatives]


def _sample_base_instance(config):
    """Sample a broad theorem schema and verify its intended proof once."""
    strategies = STRATEGIES if getattr(config, "use_mathlib", True) else CORE_STRATEGIES
    for _ in range(50):
        inst = random.choice(strategies)(config)
        if not inst or not _safe(inst.goal):
            continue
        header = _render(inst)
        runner = get_runner(use_mathlib=getattr(config, "use_mathlib", True))
        if not _safe(inst.primary) or not runner.check(header + "  " + inst.primary + "\n")[0]:
            continue
        inst.header = header
        inst.use_mathlib = getattr(config, "use_mathlib", True)
        return inst
    raise RuntimeError("failed to produce a Lean theorem instance")


def make_instance(config):
    """Build a verified proof and closely corrupted nonproof candidates."""
    n_cand = min(5, max(2, int(getattr(config, "n_candidates", 4))))
    for _ in range(50):
        inst = _sample_base_instance(config)
        local = [f"exact {name}" for name, _ in inst.hyps]
        proposed = [*_mutations(inst.primary), *_contextual_mutations(inst.primary, inst.header), *local]
        runner = get_runner(use_mathlib=inst.use_mathlib)
        negatives = _ranked_noncompiling(
            inst.primary, proposed, inst.header + "  __ANSWER__\n", runner,
            limit=n_cand - 1,
        )
        if not negatives:
            continue
        candidates = [inst.primary, *negatives[:n_cand - 1]]
        labels = [True] + [False] * (len(candidates) - 1)
        order = list(range(len(candidates)))
        random.shuffle(order)
        return edict(
            kind=inst.kind, header=inst.header,
            primary=inst.primary, elegant=inst.primary,
            candidates=[candidates[i] for i in order],
            labels=[labels[i] for i in order],
            use_mathlib=inst.use_mathlib,
        )
    raise RuntimeError("failed to produce a Lean instance")


# ============================================================================
# Multi-line proof scripts for line-level SFT tasks
# ============================================================================

def _script_code(header, lines):
    return header + "".join(f"  {line}\n" for line in lines)


EASY_MISSING_LINES = frozenset({
    "aesop", "assumption", "decide", "linarith", "norm_num", "omega",
    "rfl", "ring", "ring_nf", "simp", "simp_all", "tauto",
})
AUTOMATION_TACTICS = frozenset({
    "aesop", "decide", "linarith", "norm_num", "omega", "ring", "ring_nf",
    "simp", "simp_all", "tauto",
})


def _is_easy_missing_line(line):
    s = str(line).strip()
    return s in EASY_MISSING_LINES or bool(re.fullmatch(r"[A-Za-z_][\w.']*", s))


def _uses_automation(line):
    return bool(set(re.findall(r"\b[A-Za-z_][\w']*\b", str(line))) & AUTOMATION_TACTICS)


_MUTATION_PAIRS = (
    ("le_trans", "lt_trans"),
    ("lt_of_lt_of_le", "lt_of_le_of_lt"),
    ("Int.le_trans", "Int.lt_trans"),
    ("Int.lt_of_lt_of_le", "Int.lt_of_le_of_lt"),
    ("add_le_add", "add_lt_add"),
    ("add_le_add_left", "add_le_add_right"),
    ("dvd_add", "dvd_sub"),
    ("dvd_mul_of_dvd_left", "dvd_mul_of_dvd_right"),
    ("abs_nonneg", "sq_nonneg"),
    ("sq_nonneg", "mul_self_nonneg"),
    ("inf_sup_left", "inf_sup_right"),
    ("sup_inf_left", "sup_inf_right"),
    ("inf_assoc", "sup_assoc"),
    ("Set.inter_subset_inter", "Set.union_subset_union"),
    ("Or.inl", "Or.inr"),
)


def _swap_token(line, a, b):
    pattern = re.compile(rf"(?<![\w.])({re.escape(a)}|{re.escape(b)})(?![\w.])")
    return pattern.sub(lambda m: b if m.group(1) == a else a, line)


def _swap_projections(line):
    return re.sub(r"\.(1|2)(?!\d)", lambda m: ".2" if m.group(1) == "1" else ".1", line)


def _argument_mutations(line):
    proof_head = r"[A-Za-z_][\w.']*(?:\.[A-Za-z_][\w.']*)*"
    patterns = (
        rf"^((?:.*?\b)?(?:exact|using)\s+{proof_head})\s+(.+)$",
        rf"^(.+?:=\s*{proof_head})\s+(.+)$",
    )
    for pattern in patterns:
        m = re.match(pattern, line)
        if not m:
            continue
        prefix, arg_text = m.group(1), m.group(2)
        args = arg_text.split()
        if len(args) < 2 or any(not re.fullmatch(r"[A-Za-z_][\w.']*", arg) for arg in args):
            return []
        variants = []
        swaps = [(len(args) - 2, len(args) - 1)]
        if len(args) > 2:
            swaps.extend([(0, 1), (0, len(args) - 1)])
        for i, j in swaps:
            mutated = list(args)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            variants.append(f"{prefix} {' '.join(mutated)}")
        if len(args) == 3:
            variants.append(f"{prefix} {' '.join((args[1], args[2], args[0]))}")
        for i in range(len(args)):
            variants.append(f"{prefix} {' '.join(args[:i] + args[i + 1:])}")
            variants.append(f"{prefix} {' '.join(args[:i] + [args[i]] + args[i:])}")
        return variants
    return []


def _numbered_name_mutations(line):
    m = re.match(r"^(have\s+\w+\s*:\s*.+?:=\s*)(.+)$", line)
    if not m:
        return []
    prefix, rhs = m.groups()
    variants = []
    for tok in re.finditer(r"\b(hp|h)(\d+)\b", rhs):
        stem, n_text = tok.group(1), tok.group(2)
        n = int(n_text)
        for delta in (-1, 1):
            if n + delta < 0:
                continue
            mutated = rhs[:tok.start()] + f"{stem}{n + delta}" + rhs[tok.end():]
            variants.append(prefix + mutated)
    return variants


def _mutations(line):
    seen = set()

    def add(candidate):
        if candidate != line and candidate not in seen:
            seen.add(candidate)
            yield candidate

    for a, b in _MUTATION_PAIRS:
        swapped = _swap_token(line, a, b)
        yield from add(swapped)
    projected = _swap_projections(line)
    yield from add(projected)
    for candidate in _argument_mutations(line):
        yield from add(candidate)
    for candidate in _numbered_name_mutations(line):
        yield from add(candidate)


def _contextual_mutations(line, context):
    """Substitute identifiers that are locally available and lexically comparable."""
    groups = (
        r"\bhp\d+\b", r"\bh\d+\b", r"\bp\d+\b",
        r"\b[a-n]\b", r"\b[s-z]\b",
    )
    out = []
    for pattern in groups:
        local = sorted(set(re.findall(pattern, context)))
        for match in re.finditer(pattern, line):
            for replacement in local:
                if replacement != match.group():
                    out.append(line[:match.start()] + replacement + line[match.end():])
    for match in re.finditer(r"\b\d+\b", line):
        n = int(match.group())
        for replacement in {max(0, n - 1), n + 1}:
            out.append(line[:match.start()] + str(replacement) + line[match.end():])
    return out


def _ranked_noncompiling(answer, candidates, template, runner, limit=None):
    seen, ranked = set(), []
    for candidate in candidates:
        candidate = str(candidate).strip()
        if (candidate == answer or candidate in seen or not _safe(candidate)
                or _uses_automation(candidate)):
            continue
        seen.add(candidate)
        similarity = SequenceMatcher(None, answer, candidate).ratio()
        ranked.append((similarity, random.random(), candidate))
    ranked.sort(reverse=True)
    valid = []
    for _, _, candidate in ranked:
        if not runner.check(template.replace("__ANSWER__", candidate))[0]:
            valid.append(candidate)
            if limit is not None and len(valid) >= limit:
                break
    return valid


def _line_options(lines, answer, max_options=6, template=None, runner=None, extra=()):
    max_options = max(2, int(max_options))
    context = template or "\n".join(lines)
    candidates = [
        *list(_mutations(answer)),
        *_contextual_mutations(answer, context),
        *extra,
        *lines,
    ]
    negatives = _ranked_noncompiling(
        answer, candidates, template, runner, limit=max_options - 1,
    )
    options = negatives[:max_options - 1]
    if len(options) < max_options - 1:
        return []
    options.insert(random.randrange(max_options), answer)
    return options


def _implicit_line_space(line, context, max_candidates=16, min_slots=1):
    """A small Cartesian grammar over locally available identifiers."""
    rhs = line.find(":=") + 2 if ":=" in line else (len("exact ") if line.startswith("exact ") else 0)
    groups = (r"\bhp\d+\b", r"\bh\d+\b", r"\bp\d+\b", r"\b[a-n]\b", r"\b[s-z]\b")
    choices = []
    for pattern in groups:
        domain = sorted(set(re.findall(pattern, context)))
        if len(domain) < 2:
            continue
        for match in re.finditer(pattern, line):
            if match.start() >= rhs:
                values = [match.group(), *random.sample(
                    [x for x in domain if x != match.group()],
                    min(3, len(domain) - 1),
                )]
                random.shuffle(values)
                choices.append((match.start(), match.end(), values))
    random.shuffle(choices)
    slots, size = [], 1
    for choice in choices:
        if any(choice[0] < end and start < choice[1] for start, end, _ in slots):
            continue
        if size * len(choice[2]) <= max_candidates:
            slots.append(choice)
            size *= len(choice[2])
        if len(slots) == 3:
            break
    if len(slots) < min_slots:
        return None
    slots.sort()

    def fill(values):
        out, offset = line, 0
        for (start, end, _), value in zip(slots, values):
            out = out[:start + offset] + value + out[end + offset:]
            offset += len(value) - (end - start)
        return out

    candidates = list(dict.fromkeys(fill(values) for values in product(*(s[2] for s in slots))))
    form, offset, rules = line, 0, []
    for i, (start, end, values) in enumerate(slots, 1):
        name = f"X{i}"
        form = form[:start + offset] + name + form[end + offset:]
        offset += len(name) - (end - start)
        rules.append(f"{name} := {' | '.join(values)}")
    return form, rules, candidates


def _valid_missing_indices(lines, level=0):
    candidates = [
        i for i, line in enumerate(lines)
        if not (level >= 2 and len(lines) > 1 and i == len(lines) - 1)
        and not _is_easy_missing_line(line)
    ]
    if not candidates:
        return []
    primary_prefixes = ("have ", "rcases ", "cases ", "exact ")
    preferred = [i for i in candidates if lines[i].startswith(primary_prefixes)]
    if preferred:
        return preferred
    intro = [i for i in candidates if lines[i].startswith("intro ")]
    return intro or candidates


def _proof_script_set_union(config):
    names = list("stuvwxyz")[: max(3, min(6, int(config.n_vars) + 1))]
    s, t, u = random.sample(names, 3)
    if random.random() < 0.5:
        goal = f"{s} ∪ {u} ⊆ {t} ∪ {u}"
        lines = ["exact Set.union_subset_union h0 subset_rfl"]
    else:
        goal = f"{u} ∪ {s} ⊆ {u} ∪ {t}"
        lines = ["exact Set.union_subset_union subset_rfl h0"]
    return edict(
        kind="proof_script:set_union",
        header=_render(edict(decl=f"({' '.join(names)} : Set Int)",
                             hyps=[("h0", f"{s} ⊆ {t}")], goal=goal)),
        lines=lines,
    )


def _proof_script_set_inter(config):
    names = list("stuvwxyz")[: max(3, min(6, int(config.n_vars) + 1))]
    s, t, u = random.sample(names, 3)
    if random.random() < 0.5:
        goal = f"{s} ∩ {u} ⊆ {t} ∩ {u}"
        line = "exact Set.inter_subset_inter h0 subset_rfl"
    else:
        goal = f"{u} ∩ {s} ⊆ {u} ∩ {t}"
        line = "exact Set.inter_subset_inter subset_rfl h0"
    return edict(
        kind="proof_script:set_inter",
        header=_render(edict(decl=f"({' '.join(names)} : Set Int)",
                             hyps=[("h0", f"{s} ⊆ {t}")], goal=goal)),
        lines=[line],
    )


def _proof_script_order(config):
    n_hyps = max(3, min(10, int(config.n_hyps) + 1))
    names = _vars(n_hyps + 1)
    hyps = [(f"h{i}", f"{names[i]} ≤ {names[i + 1]}") for i in range(n_hyps)]
    goal = f"{names[0]} ≤ {names[-1]}"
    if n_hyps <= 3 and random.random() < 0.4:
        proof = f"h{n_hyps - 1}"
        for i in reversed(range(n_hyps - 1)):
            proof = f"(le_trans h{i} {proof})"
        lines = [f"exact {proof}"]
    else:
        lines = []
        prev = "h0"
        for i in range(1, n_hyps):
            name = f"h{n_hyps + i - 1}"
            lines.append(f"have {name} : {names[0]} ≤ {names[i + 1]} := le_trans {prev} h{i}")
            prev = name
        lines.append(f"exact {prev}")
    return edict(
        kind="proof_script:order",
        header=_render(edict(decl=f"({' '.join(names)} : Int)", hyps=hyps, goal=goal)),
        lines=lines,
    )


def _proof_script_prop(config):
    n_hyps = max(2, min(10, int(config.n_hyps)))
    props = [f"p{i}" for i in range(n_hyps + 1)]
    hyps = [(f"h{i}", f"{props[i]} → {props[i + 1]}") for i in range(n_hyps)]
    lines = ["intro hp"]
    prev = "hp"
    for i in range(n_hyps):
        name = f"hp{i + 1}"
        lines.append(f"have {name} : {props[i + 1]} := h{i} {prev}")
        prev = name
    lines.append(f"exact {prev}")
    return edict(
        kind="proof_script:prop_chain",
        header=_render(edict(decl=f"({' '.join(props)} : Prop)",
                             hyps=hyps, goal=f"{props[0]} → {props[-1]}")),
        lines=lines,
    )


def _proof_script_finset(config):
    names = list("stuvwxyz")[: max(3, min(6, int(config.n_vars) + 1))]
    a, b, c = random.sample(names, 3)
    templates = (
        (f"{a} ∩ ({b} ∪ {c}) = ({a} ∩ {b}) ∪ ({a} ∩ {c})", f"simpa using inf_sup_left {a} {b} {c}"),
        (f"({a} ∪ {b}) ∩ {c} = ({a} ∩ {c}) ∪ ({b} ∩ {c})", f"simpa using inf_sup_right {a} {b} {c}"),
        (f"{a} ∪ ({b} ∩ {c}) = ({a} ∪ {b}) ∩ ({a} ∪ {c})", f"simpa using sup_inf_left {a} {b} {c}"),
    )
    goal, proof = random.choice(templates)
    return edict(
        kind="proof_script:finset",
        header=_render(edict(decl=f"({' '.join(names)} : Finset Nat)", hyps=[], goal=goal)),
        lines=[proof],
    )


def _proof_script_core(config):
    n_hyps = max(2, min(6, int(getattr(config, "n_hyps", 2))))
    props = [f"p{i}" for i in range(n_hyps + 1)]
    hyps = [(f"h{i}", f"{props[i]} → {props[i + 1]}") for i in range(n_hyps)]
    lines = ["intro hp"]
    prev = "hp"
    for i in range(n_hyps):
        name = f"hp{i + 1}"
        lines.append(f"have {name} : {props[i + 1]} := h{i} {prev}")
        prev = name
    lines.append(f"exact {prev}")
    return edict(
        kind="core_script:prop_chain",
        header=_render(edict(decl=f"({' '.join(props)} : Prop)",
                             hyps=hyps, goal=f"{props[0]} → {props[-1]}")),
        lines=lines,
    )


def _proof_script_forward_order(config):
    try:
        fg = gen_forward_order_graph(config)
    except RuntimeError:
        return None
    if fg is None:
        return None
    return edict(
        kind="proof_script:forward_order",
        header=_render(edict(decl=fg.decl, hyps=fg.leaf_hyps, goal=fg.goal)),
        lines=fg.proof.splitlines(),
    )


def _proof_script_instance(config):
    """Reuse the broad theorem schemas while keeping the completion to one line."""
    inst = _sample_base_instance(config)
    if "\n" in inst.primary:
        return None
    local = [f"exact {name}" for name, _ in inst.hyps]
    return edict(
        kind=f"proof_script:{inst.kind}",
        header=inst.header,
        lines=[inst.primary],
        candidate_lines=local,
    )


_PROOF_SCRIPT_BUILDERS = (
    _proof_script_set_union,
    _proof_script_set_inter,
    _proof_script_order,
    _proof_script_prop,
    _proof_script_finset,
)


def make_proof_script(config):
    level = int(getattr(config, "level", 0))
    if not getattr(config, "use_mathlib", True):
        for _ in range(50):
            builder = random.choice((_proof_script_core, _proof_script_instance))
            script = builder(config)
            if script is None:
                continue
            if get_runner(use_mathlib=False).check(_script_code(script.header, script.lines))[0]:
                return script
        raise RuntimeError("failed to produce a Lean proof script")
    weighted_builders = [
        (_proof_script_set_union, 1),
        (_proof_script_set_inter, 1),
        (_proof_script_finset, 1),
        (_proof_script_order, 1 + level),
        (_proof_script_prop, 1 + level),
        (_proof_script_instance, 4),
    ]
    forward_weight = max(0, 6 * (level - 1))
    if forward_weight:
        weighted_builders.append((_proof_script_forward_order, forward_weight))
    builders, weights = zip(*weighted_builders)
    for _ in range(50):
        script = random.choices(builders, weights=weights, k=1)[0](config)
        if script is None:
            continue
        if all(_safe(line) for line in script.lines) and get_runner().check(_script_code(script.header, script.lines))[0]:
            return script
    raise RuntimeError("failed to produce a Lean proof script")


# ============================================================================
# Lean-verified forward derivation graphs
# ============================================================================


def _split_order_formula(formula):
    m = re.fullmatch(r"\s*(.+?)\s*(≤|<)\s*(.+?)\s*", formula)
    if not m:
        return None
    return m.group(1).strip(), m.group(2), m.group(3).strip()


def _lean_example(decl, hyps, goal, proof):
    hyp_str = " ".join(f"({h} : {b})" for h, b in hyps)
    decl = decl + " " if decl else ""
    return f"example {decl}{hyp_str} : {goal} := by\n  exact {proof}\n"


def _node_data(G, node):
    return G.nodes[node]["data"]


def _leaf_nodes(G):
    return [n for n in nx.topological_sort(G) if G.in_degree(n) == 0]


def _leaf_ancestor_nodes(G, target_node):
    ancestors = nx.ancestors(G, target_node)
    return [n for n in _leaf_nodes(G) if n in ancestors]


def _add_lean_node(G, node):
    G.add_node(node.clause_id, data=node)
    for parent in node.parents:
        G.add_edge(parent, node.clause_id)


def _fresh_clause_id(G):
    return f"c{len(G.nodes)}"


def _rule_application(G, left, right, decl, runner):
    lf = _split_order_formula(_node_data(G, left).clause_formula)
    rf = _split_order_formula(_node_data(G, right).clause_formula)
    if not lf or not rf or lf[2] != rf[0]:
        return None
    rule = _ORDER_RULES.get((lf[1], rf[1]))
    if rule is None:
        return None
    theorem, out_op = rule
    conclusion = f"{lf[0]} {out_op} {rf[2]}"
    if any(_node_data(G, n).clause_formula == conclusion for n in G.nodes):
        return None
    proof = f"{theorem} p0 p1"
    code = _lean_example(
        decl,
        [("p0", _node_data(G, left).clause_formula),
         ("p1", _node_data(G, right).clause_formula)],
        conclusion,
        proof,
    )
    ok, _ = runner.check(code)
    if not ok:
        return None
    depth = max(_node_data(G, left).depth, _node_data(G, right).depth) + 1
    clause_id = _fresh_clause_id(G)
    return LeanDerivationNode(
        clause_id=clause_id,
        clause_formula=conclusion,
        parents=(left, right),
        inference=theorem,
        role="plain",
        interesting_score=float(depth),
        full_cnf_clause=f"cnf({clause_id},plain,{conclusion})",
        proof=f"{theorem} {{0}} {{1}}",
        depth=depth,
    )


def _hypothesis_names(G):
    leaves = [n for n in nx.topological_sort(G) if G.in_degree(n) == 0]
    return {n: f"h{i}" for i, n in enumerate(leaves)}


def make_lean_have_chain(G, target_node, decl="", leaf_hyp_names=None):
    sub = G.subgraph(nx.ancestors(G, target_node) | {target_node}).copy()
    leaf_hyp_names = dict(leaf_hyp_names or _hypothesis_names(sub))
    node_to_name = dict(leaf_hyp_names)
    lines = []
    next_idx = len(leaf_hyp_names)

    for node in nx.topological_sort(sub):
        if node in node_to_name:
            continue
        data = _node_data(sub, node)
        parents = list(data.parents) if data.parents else list(sub.predecessors(node))
        parent_names = [node_to_name[p] for p in parents]
        name = f"h{next_idx}"
        next_idx += 1
        proof = data.proof.format(*parent_names) if data.proof else f"{data.inference} {' '.join(parent_names)}"
        lines.append(f"have {name} : {data.clause_formula} := {proof}")
        node_to_name[node] = name

    lines.append(f"exact {node_to_name[target_node]}")
    proof = "\n".join(lines)
    refs = set(re.findall(r"\bh\d+\b", proof))
    defined = set(leaf_hyp_names.values())
    for line in lines:
        m = re.match(r"have\s+(h\d+)\s*:", line)
        if m:
            defined.add(m.group(1))
    if not refs <= defined:
        raise RuntimeError(f"undefined Lean proof references: {sorted(refs - defined)}")
    return proof


def _render_forward_theorem(decl, leaf_hyps, goal, proof_body, name="ex"):
    hyp_str = " ".join(f"({h} : {b})" for h, b in leaf_hyps)
    args = " ".join(x for x in (decl, hyp_str) if x)
    args = f" {args}" if args else ""
    body = "\n".join(f"  {line}" for line in proof_body.splitlines())
    return f"theorem {name}{args} : {goal} := by\n{body}\n"


def _cheap_solvers(header, use_mathlib):
    candidates = ("omega", "simp", "aesop", "tauto", "ring") if use_mathlib else ("simp", "tauto", "rfl")
    runner = get_runner(use_mathlib=use_mathlib)
    solved = []
    for tactic in candidates:
        if not _safe(tactic):
            continue
        ok, _ = runner.check(header + f"  {tactic}\n")
        if ok:
            solved.append(tactic)
    return solved


def gen_forward_order_graph(config):
    n_edges = max(6, min(9, int(getattr(config, "n_hyps", 4)) + 2))
    n_vars = n_edges + 1
    names = _vars(n_vars)
    decl = f"({' '.join(names)} : Int)"
    G = nx.DiGraph()
    strict_count = 0

    for i in range(n_edges):
        op = "<" if random.random() < 0.35 else "≤"
        strict_count += int(op == "<")
        formula = f"{names[i]} {op} {names[i + 1]}"
        clause_id = f"p{i}"
        _add_lean_node(G, LeanDerivationNode(
            clause_id=clause_id,
            clause_formula=formula,
            inference="hypothesis",
            role="axiom",
            interesting_score=0.0,
            full_cnf_clause=f"cnf({clause_id},axiom,{formula})",
            proof=f"h{i}",
            depth=0,
        ))

    if strict_count == n_edges:
        data = _node_data(G, f"p{random.randrange(n_edges)}")
        data.clause_formula = data.clause_formula.replace("<", "≤", 1)

    runner = get_runner(use_mathlib=getattr(config, "use_mathlib", True))
    max_depth = max(3, min(6, int(getattr(config, "expr_depth", 4)) + 1))
    accepted = 0
    proposals = 0
    frontier = "p0"
    for i in range(1, n_edges):
        proposals += 1
        node = _rule_application(G, frontier, f"p{i}", decl, runner)
        if node is None or node.depth > max_depth:
            break
        frontier = node.clause_id
        _add_lean_node(G, node)
        accepted += 1

    for left, right in (("p0", "p1"), ("p2", "p3"), ("p4", "p5")):
        if left in G and right in G:
            proposals += 1
            node = _rule_application(G, left, right, decl, runner)
            if node is not None and node.depth <= max_depth:
                _add_lean_node(G, node)
                accepted += 1

    candidates = [
        n for n in G.nodes
        if G.in_degree(n) > 0
        and _node_data(G, n).depth >= 3
        and len(_leaf_ancestor_nodes(G, n)) >= 2
    ]
    random.shuffle(candidates)
    for target in sorted(candidates, key=lambda n: _node_data(G, n).depth, reverse=True):
        useful_leaf_nodes = _leaf_ancestor_nodes(G, target)
        all_leaf_nodes = _leaf_nodes(G)
        if (
            len(useful_leaf_nodes) < 2
            or len(useful_leaf_nodes) >= len(all_leaf_nodes)
            or _node_data(G, target).depth < 3
            or len(all_leaf_nodes) - len(useful_leaf_nodes) < 2
        ):
            continue
        leaf_names = _hypothesis_names(G)
        leaf_hyps = [(leaf_names[n], _node_data(G, n).clause_formula) for n in all_leaf_nodes]
        goal = _node_data(G, target).clause_formula
        if goal in [h for _, h in leaf_hyps]:
            continue
        used = _used_int_vars(goal, *[h for _, h in leaf_hyps])
        theorem_decl = f"({' '.join(used)} : Int)" if used else decl
        proof = make_lean_have_chain(G, target, decl=decl, leaf_hyp_names=leaf_names)
        theorem = _render_forward_theorem(theorem_decl, leaf_hyps, goal, proof)
        if not _safe(theorem):
            continue
        ok, diag = runner.check(theorem)
        if not ok:
            continue
        header = _render_forward_theorem(theorem_decl, leaf_hyps, goal, "", name="ex").rsplit(":= by\n", 1)[0] + ":= by\n"
        cheap = _cheap_solvers(header, getattr(config, "use_mathlib", True))
        return edict(
            G=G,
            target_node=target,
            decl=theorem_decl,
            leaf_hyp_names=leaf_names,
            leaf_hyps=leaf_hyps,
            goal=goal,
            proof=proof,
            theorem=theorem,
            cheap_solvers=cheap,
            stats=edict(
                proposals=proposals,
                accepted=accepted,
                acceptance_rate=(accepted / proposals if proposals else 0.0),
                proof_depth=_node_data(G, target).depth,
                useful_premises=len(useful_leaf_nodes),
                total_premises=len(all_leaf_nodes),
                distractor_premises=len(all_leaf_nodes) - len(useful_leaf_nodes),
                final_verifies=True,
                diag=diag,
            ),
        )
    return None


# ============================================================================
# Tasks
# ============================================================================

class LeanMissingLine(Task):
    """Synthesize the unique line in a small, exhaustively checked term grammar."""
    summary = "Complete a Lean proof with a uniquely valid constrained proof line."

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config or LeanConfig(use_mathlib=_profile_ready(use_mathlib=True)), timeout=120, **kwargs)

    def generate_entry(self):
        use_mathlib = getattr(self.config, "use_mathlib", True)
        runner = get_runner(use_mathlib=use_mathlib)
        level = int(getattr(self.config, "level", 0))
        multiple_choice = random.random() < self.config.multiple_choice_prob
        for _ in range(30):
            script = make_proof_script(self.config)
            valid_indices = _valid_missing_indices(script.lines, level)
            if not valid_indices:
                continue
            idx = random.choice(valid_indices)
            correct_line = script.lines[idx]
            if _is_easy_missing_line(correct_line) or _uses_automation(correct_line):
                continue
            template = script.header + "".join(
                "  __ANSWER__\n" if i == idx else f"  {line}\n"
                for i, line in enumerate(script.lines)
            )
            if multiple_choice:
                available = _line_options(
                    script.lines, correct_line, self.config.n_candidates,
                    template, runner, getattr(script, "candidate_lines", ()),
                )
                if len(available) != max(2, int(self.config.n_candidates)):
                    continue
                correct_index = available.index(correct_line) + 1
                return Entry(
                    edict(
                        kind=script.kind, template=template, available_lines=available,
                        compiling_indices=[correct_index],
                        compiling_lines=[correct_line], correct_line=correct_line,
                        correct_index=correct_index,
                        missing_line=idx + 1, use_mathlib=use_mathlib,
                        used_mathlib=use_mathlib, multiple_choice=True,
                    ),
                    str(correct_index),
                )
            space = _implicit_line_space(
                correct_line,
                template,
                max_candidates=min(16, max(4, int(self.config.n_candidates) * 2)),
                min_slots=2 if level >= 2 else 1,
            )
            if space is None:
                continue
            answer_form, slot_rules, available = space
            candidate_results = [
                runner.check(template.replace("__ANSWER__", candidate))[0]
                for candidate in available
            ]
            compiling = [line for line, ok in zip(available, candidate_results) if ok]
            if compiling != [correct_line]:
                continue
            return Entry(
                edict(
                    kind=script.kind,
                    template=template,
                    answer_form=answer_form,
                    slot_rules=slot_rules,
                    candidate_lines=available,
                    candidate_compiles=candidate_results,
                    compiling_lines=compiling,
                    n_slots=len(slot_rules),
                    n_candidates=len(available),
                    finite_space_exhaustively_checked=True,
                    correct_line=correct_line,
                    missing_line=idx + 1,
                    use_mathlib=use_mathlib,
                    used_mathlib=use_mathlib,
                    multiple_choice=False,
                ),
                correct_line,
            )
        raise RuntimeError("failed to generate a unique constrained Lean line")

    def render_prompt(self, metadata):
        imports = "Mathlib is imported." if _mget(metadata, "use_mathlib") else "Only Lean/Std is imported."
        if _mget(metadata, "multiple_choice"):
            options = "\n".join(
                f"{i}. {line}" for i, line in enumerate(_mget(metadata, "available_lines"), 1)
            )
            return (
                f"Which listed Lean proof line fills `__ANSWER__`? {imports}\n\n"
                f"THEOREM:\n{_mget(metadata, 'template')}\n"
                f"LINES:\n{options}\nAnswer with the line number."
            )
        rules = "\n".join(_mget(metadata, "slot_rules"))
        return (
            f"Fill `__ANSWER__` with a Lean proof line. {imports}\n\n"
            f"THEOREM:\n{_mget(metadata, 'template')}\n"
            f"The answer must have the form:\n{_mget(metadata, 'answer_form')}\n"
            f"where:\n{rules}\n"
            "Answer with the complete Lean line."
        )

    def score_answer(self, answer, entry):
        value = str(answer).strip().strip("`").strip()
        return float(value == str(entry.answer))


class LeanCandidateCompilation(Task):
    """True/False on whether a single candidate proof body closes the theorem."""
    summary = "Determine if a candidate proof body successfully closes a theorem in Lean."

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config or LeanConfig(), timeout=120, **kwargs)

    def generate_entry(self):
        for _ in range(20):
            inst = make_instance(self.config)
            pairs = _hard_candidate_pairs(inst)
            if pairs:
                break
        else:
            positives = [c for c, ok in zip(inst.candidates, inst.labels) if ok]
            negatives = [c for c, ok in zip(inst.candidates, inst.labels) if not ok]
            pairs = [(positive, negative) for positive in positives for negative in negatives]
            if not pairs:
                raise RuntimeError("failed to generate paired Lean proof candidates")
        want_positive = random.random() < 0.5
        positive, negative = random.choice(pairs)
        candidate = positive if want_positive else negative
        return Entry(
            edict(kind=inst.kind,
                  theorem=inst.header + "  ?\n",
                  candidate=candidate,
                  paired_candidate=negative if want_positive else positive,
                  candidate_similarity=SequenceMatcher(None, positive, negative).ratio(),
                  candidate_count=len(pairs),
                  use_mathlib=inst.use_mathlib),
            "True" if want_positive else "False",
        )

    def render_prompt(self, metadata):
        return (
            "Does this Lean 4 tactic body close the theorem?\n"
            "The answer is True or False.\n\n"
            f"THEOREM:\n{_mget(metadata, 'theorem')}\n"
            f"CANDIDATE:\n{_mget(metadata, 'candidate')}"
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip().strip("`").lower() == entry.answer.lower())
