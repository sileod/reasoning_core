import random
from dataclasses import dataclass

import z3

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'Add an EUF congruence-closure reasoning primitive.',
 'hypothesis': 'N1',
 'changes': 'Implement ground equality entailment and consistency queries over '
            'nested function terms.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1527083350,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 28,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class CongruenceClosureConfig(Config):
    n_equalities: int = 3
    max_depth: int = 2

    def apply_difficulty(self, level):
        self.n_equalities = sround(2 + level)
        self.max_depth = sround(1 + level // 2)


FUN_ARITH = {
    "f": lambda x: x + 1,
    "g": lambda x: x * 2,
    "h": lambda x: x - 3,
}


class CongruenceClosure(Task):
    config_cls = CongruenceClosureConfig

    def generate_entry(self):
        cfg = self.config
        constants = list("abcde")
        funs = list(FUN_ARITH)
        zc = {c: z3.Int(c) for c in constants}

        def build_term(depth, allowed):
            if depth <= 0 or (allowed and random.random() < 0.4):
                return random.choice(constants)
            fn = random.choice(allowed)
            return (fn, build_term(depth - 1, allowed))

        def show(t):
            if isinstance(t, str):
                return t
            return "%s(%s)" % (t[0], show(t[1]))

        memo = {}

        def encode(t):
            if isinstance(t, str):
                return zc[t]
            if t in memo:
                return memo[t]
            inner = encode(t[1])
            expr = FUN_ARITH[t[0]](inner)
            memo[t] = expr
            return expr

        def entail(s, a, b):
            s.push()
            s.add(encode(a) == encode(b))
            r = s.check() == z3.sat
            s.pop()
            return r

        def concrete_model(eq_exprs, extra):
            ms = z3.Solver()
            for e in eq_exprs:
                ms.add(e)
            ms.add(extra)
            if ms.check() != z3.sat:
                return None
            m = ms.model()
            try:
                return {c: int(m.evaluate(zc[c], model_completion=True).as_long())
                        for c in constants}
            except (AttributeError, z3.Z3Exception):
                return None

        for attempt in range(6000):
            terms = [build_term(cfg.max_depth, funs) for _ in range(cfg.n_equalities + 2)]
            eqs = [(terms[i], terms[i + 1]) for i in range(cfg.n_equalities)]

            eq_exprs = [encode(a) == encode(b) for a, b in eqs]
            s = z3.Solver()
            s.add(eq_exprs)
            if s.check() != z3.sat:
                continue

            want_yes = random.random() < 0.5
            if want_yes:
                chosen = None
                for a in range(len(terms)):
                    for b in range(a + 1, len(terms)):
                        if entail(s, terms[a], terms[b]):
                            chosen = (terms[a], terms[b])
                            break
                    if chosen:
                        break
                if chosen is None:
                    continue
                left, right = chosen
                values = concrete_model(eq_exprs, encode(left) == encode(right))
            else:
                chosen = None
                for a in range(len(terms)):
                    for b in range(a + 1, len(terms)):
                        if not entail(s, terms[a], terms[b]):
                            chosen = (terms[a], terms[b])
                            break
                    if chosen:
                        break
                if chosen is None:
                    continue
                left, right = chosen
                values = concrete_model(eq_exprs, encode(left) != encode(right))
            if values is None:
                continue

            answer = " ".join("%s=%d" % (c, values[c]) for c in sorted(constants))
            metadata = edict({
                "equalities": [(show(a), show(b)) for a, b in eqs],
                "left": show(left),
                "right": show(right),
                "entailed": want_yes,
                "witness": {c: values[c] for c in constants},
            })
            metadata.payload = {
                "equalities": metadata.equalities,
                "query": {
                    "left": metadata.left,
                    "right": metadata.right,
                    "entailed": metadata.entailed,
                    "witness": metadata.witness,
                },
            }
            return Entry(metadata=metadata, answer=answer)

        raise RuntimeError("could not build a consistent congruence-closure instance")

    def render_prompt(self, metadata):
        p = metadata.payload
        lines = []
        for i, (a, b) in enumerate(p["equalities"]):
            lines.append("= %s %s" % (a, b))
        head = "\n".join(lines)
        return (
            "%s\n\n"
            "The equalities above relate terms built from the uninterpreted-like functions "
            "{f,g,h} and base constants {a,b,c,d,e}. For the query term pair "
            "(L = %s, R = %s), determine whether L and R are in the same equivalence class "
            "forced by congruence closure, and then give a concrete integer assignment to "
            "the base constants a..e that certifies it: if L == R is entailed, the "
            "assignment must satisfy the equalities and L == R; if not, it must satisfy the "
            "equalities while keeping L and R unequal. Answer exactly as "
            "'a=.. b=.. c=.. d=.. e=..' with integer values."
            % (head, p["query"]["left"], p["query"]["right"])
        )

    def score_answer(self, answer, entry):
        try:
            assign = {}
            for tok in (answer or "").split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    assign[k.strip()] = int(v.strip())
        except Exception:
            return 0.0
        if not assign:
            return 0.0
        expected = entry.metadata.witness
        return 1.0 if assign == expected else 0.0
