from dataclasses import dataclass
import random

from easydict import EasyDict as edict

from reasoning_core.template import Config, Entry, Task, render_payload, stochastic_rounding as sround
from reasoning_core.tasks.logic_depth import (
    Atom,
    Rule,
    Theory,
    chase,
    PredSig,
    _domain_pack,
    _focused_theory,
    render,
    trace_for,
    atom_text,
)


@dataclass
class ConjunctiveLogicConfig(Config):
    n_entities: int = 4
    n_unary_preds: int = 12
    n_rules: int = 6
    max_depth: int = 3
    n_distractors: int = 1
    domain_packs: tuple = ("surface", "abstract")

    def apply_difficulty(self, level):
        self.n_entities = min(7, int(self.n_entities) + level)
        self.n_rules += level
        self.max_depth = min(4, int(self.max_depth) + (1 if level >= 3 else 0))
        self.n_distractors += 2 * level


GENERIC_UNARY = (
    "approved", "careful", "trained", "trusted", "active", "verified",
    "alert", "eligible", "quiet", "skilled", "reliable", "cleared",
)


class ConjunctiveLogicQA(Task):
    summary = "Multi-step conjunctive queries answered only via two independent derivations."

    def __init__(self, config=None):
        super().__init__(config=config or ConjunctiveLogicConfig())
        self._case_state = {}

    def _mk_sigs(self, pack, n_entities, n_unary):
        n = min(max(3, n_entities), 7)
        if pack == "abstract":
            ents = {"entity": tuple(x.title() for x in (
                "alice", "bruno", "clara", "david", "elena", "farah", "george"))[:n]}
            names = tuple(f"p{i}" for i in range(n_unary))
            sigs = {p: PredSig(p, ("entity",)) for p in names}
        else:
            ents = {"person": ("alice", "bruno", "clara", "david", "elena", "farah", "george")[:n]}
            names = GENERIC_UNARY[:n_unary]
            sigs = {p: PredSig(p, ("person",)) for p in names}
        return ents, names, sigs

    def _build_case(self):
        cfg = self.config
        pack = random.choice(tuple(cfg.domain_packs))
        ents, names, sigs = self._mk_sigs(pack, cfg.n_entities, cfg.n_unary_preds)
        people = list(next(iter(ents.values())))

        for _ in range(400):
            if len(names) < 10:
                continue
            available = list(names)
            random.shuffle(available)
            # Branch1: a1,b1 -> h1. Branch2: a2,b2 -> h2.
            # base1->a1, base2->b1, base3->a2, base4->b2.
            base1, base2, base3, base4 = available[0:4]
            a1, b1 = available[4], available[5]
            a2, b2 = available[6], available[7]
            h1, h2 = available[8], available[9]
            # reserve remaining as exception-guard predicates
            guards = available[10:]

            x = random.choice(people)
            facts = [
                Atom(base1, (x,)),
                Atom(base2, (x,)),
                Atom(base3, (x,)),
                Atom(base4, (x,)),
            ]
            rules = [
                Rule((Atom(base1, ("?x",)),), Atom(a1, ("?x",)), "cj", "u_imp"),
                Rule((Atom(base2, ("?x",)),), Atom(b1, ("?x",)), "cj", "u_imp"),
                Rule((Atom(a1, ("?x",)), Atom(b1, ("?x",))), Atom(h1, ("?x",)), "cj", "u_and"),
                Rule((Atom(base3, ("?x",)),), Atom(a2, ("?x",)), "cj", "u_imp"),
                Rule((Atom(base4, ("?x",)),), Atom(b2, ("?x",)), "cj", "u_imp"),
                Rule((Atom(a2, ("?x",)), Atom(b2, ("?x",))), Atom(h2, ("?x",)), "cj", "u_and"),
            ]

            def depth_of(res, pred):
                atom = Atom(pred, (x,))
                if atom not in res.closure:
                    return None
                return res.derivations[atom].depth

            res = chase(Theory(facts, rules, [], sigs, ents, pack), max_depth=None)
            if res.inconsistent:
                continue
            ds_a1, ds_b1 = depth_of(res, a1), depth_of(res, b1)
            ds_a2, ds_b2 = depth_of(res, a2), depth_of(res, b2)
            for d in (ds_a1, ds_b1, ds_a2, ds_b2):
                if d is None or d < 1:
                    break
            else:
                # Each conjunct (a1,b1,a2,b2) must be non-lexical.
                if min(ds_a1, ds_b1, ds_a2, ds_b2) < 1:
                    continue
                # Cross-branch independence: removing branch1 base facts must break h1,
                # removing branch2 base facts must break h2, and neither head may be
                # derivable purely from the other branch.
                def base_facts(bases):
                    s = set(bases)
                    return [f for f in facts if f in s], [f for f in facts if f not in s]

                # Branch1 depends on base1,base2 only; branch2 on base3,base4 only.
                # Also guard against a1/b1 being accidentally derivable from each other
                # or from branch2 (would make a conjunct lexical-ish / redundant).
                missing_b1 = [f for f in facts if f not in {Atom(base1, (x,)), Atom(base2, (x,))}]
                sub1 = Theory(missing_b1, rules, [], sigs, ents, pack)
                r1 = chase(sub1, max_depth=None)
                if r1.inconsistent or Atom(h1, (x,)) in r1.closure:
                    continue
                missing_b2 = [f for f in facts if f not in {Atom(base3, (x,)), Atom(base4, (x,))}]
                sub2 = Theory(missing_b2, rules, [], sigs, ents, pack)
                r2 = chase(sub2, max_depth=None)
                if r2.inconsistent or Atom(h2, (x,)) in r2.closure:
                    continue
                break
        else:
            return None

        theory = Theory(facts, rules, [], sigs, ents, pack)
        res = chase(theory, max_depth=None)
        h1a, h2a = Atom(h1, (x,)), Atom(h2, (x,))
        return edict(theory=theory, res=res, x=x, pack=pack,
                     h1=h1a, h2=h2a,
                     conjuncts=(Atom(a1, (x,)), Atom(b1, (x,)), Atom(a2, (x,)), Atom(b2, (x,))))

    def generate_entry(self):
        for _ in range(200):
            case = self._build_case()
            if not case:
                continue
            targets = [case.h1, case.h2]
            focused = _focused_theory(case.theory, case.res, targets, self.config.n_distractors)
            fres = chase(focused, max_depth=None)
            if fres.inconsistent or any(a not in fres.closure for a in targets):
                continue
            lines, source, _ = render(focused)
            c1, c2, c3, c4 = case.conjuncts
            question = (
                f"Which entity can be shown to be both {atom_text(c1, case.pack).split(' is ', 1)[1]} "
                f"and {atom_text(c2, case.pack).split(' is ', 1)[1]}, "
                f"as well as both {atom_text(c3, case.pack).split(' is ', 1)[1]} "
                f"and {atom_text(c4, case.pack).split(' is ', 1)[1]}?"
            )
            cot = "\n\n".join(
                trace_for(a, fres.derivations, source, case.pack) for a in targets
            )
            meta = edict(
                premise=lines,
                question=question,
                domain_pack=case.pack,
                cot=cot,
            )
            meta.payload = {"premise": "\n".join(lines), "question": question}
            return Entry(meta, case.x)
        raise RuntimeError("could not generate a conjunctive logic example")

    def render_prompt(self, meta):
        return f"{render_payload(meta.payload)}\n\nThe answer is one entity name."

    def score_answer(self, answer, entry):
        truth = str(entry.answer).strip().lower()
        pred = str(answer).strip().lower().rstrip(".")
        return float(pred == truth)


TASK_META = {'parent_source_id': 'c46e36da9bb5649f33aca10cade20290e278794415f21bfd119b16ca23a4c977',
 'idea': 'Test shallow composition across separate proof branches.',
 'hypothesis': 'H10',
 'changes': 'Require conjunctive queries whose predicates derive from distinct '
            'branches.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 988849263,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 20,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
