# def_conflict_depth v1
from collections import Counter
from dataclasses import dataclass
import random

from easydict import EasyDict as edict

from reasoning_core.template import Config, Entry, Task, render_payload, stochastic_rounding as sround
from reasoning_core.tasks.logic_depth import (
    Atom,
    Not,
    PredSig,
    Rule,
    Theory,
    naf_chase,
    render_naf,
    _naf_atom_text,
    _capitalized,
    _naf_property_text,
    _naf_rule_text,
)
from reasoning_core.tasks.logic_depth import NAMES, TAG_WORDS


@dataclass
class DefeasibleConflictConfig(Config):
    conflict_depth: int = 1
    n_entities: int = 4
    block_rate: float = 0.5

    def apply_difficulty(self, level):
        self.conflict_depth = 1 + level
        self.n_entities = min(8, self.n_entities + level)


def _safe_block(exc_preds, facts):
    return {e for e in exc_preds if Atom(e, (), False) not in facts}


def _build_theory(depth, ents):
    typ = "entity"
    names = {}
    domains = []
    props = [f"p{i}" for i in range(depth + 1)]
    excs = [f"exc{i}" for i in range(1, depth + 1)]
    for p in props + excs:
        names[p] = PredSig(p, (typ,), ())
    sigs = {p: names[p] for p in props + excs}
    ents = {typ: ents}

    rules = []
    for i in range(depth):
        rules.append(Rule(
            (Atom(props[i], ("?x",)), Not(Atom(excs[i], ("?x",)))),
            Atom(props[i + 1], ("?x",)),
            "def_conflict", "default",
        ))
        rules.append(Rule(
            (Atom(excs[i], ("?x",)),),
            Atom(props[i + 1], ("?x",), False),
            "def_conflict", "exception",
        ))
    for i in range(depth):
        rules.append(Rule(
            (Atom(props[i], ("?x",), False),),
            Atom(props[i + 1], ("?x",), False),
            "def_conflict", "propagate",
        ))
    return Theory([], rules, [], sigs, ents, "abstract"), props, excs


def _outcome_label(theory, target, prop):
    res = naf_chase(theory, max_depth=None)
    if res.inconsistent:
        return None
    if Atom(prop, (target,)) in res.closure:
        return "Yes"
    if Atom(prop, (target,), False) in res.closure:
        return "No"
    return "Maybe"


class DefeasibleConflictDepth(Task):
    summary = "Defeasible NLI judged by nested default-exception conflict depth."

    def __init__(self, config=None):
        super().__init__(config=config or DefeasibleConflictConfig())
        self._state = Counter()

    def generate_entry(self):
        depth = int(self.config.conflict_depth)
        label_counts = self._state
        labels = ("Yes", "No", "Maybe")
        wanted = min(label_counts[(depth, x)] for x in labels) if label_counts else 0
        candidates = [x for x in labels if label_counts[(depth, x)] == wanted]
        target_label = random.choice(candidates)

        for _ in range(300):
            people = random.sample(list(NAMES[: self.config.n_entities]), self.config.n_entities)
            theory, props, excs = _build_theory(depth, people)
            target = people[0]
            hyp_prop = props[-1]

            facts = []
            if target_label == "Yes":
                facts.append(Atom(props[0], (target,)))
            elif target_label == "No":
                facts.append(Atom(props[0], (target,)))
                k = random.randint(1, depth)
                facts.append(Atom(excs[k - 1], (target,)))
            else:
                facts.append(Atom(excs[depth - 1], (people[1],)))

            used = set(facts)
            for ent in people[2:]:
                if random.random() < self.config.block_rate and len(used) < self.config.n_entities + 2:
                    f = Atom(excs[random.randint(1, depth) - 1], (ent,))
                    if f not in used:
                        facts.append(f)
                        used.add(f)
            theory.facts = facts

            lbl = _outcome_label(theory, target, hyp_prop)
            if lbl != target_label:
                continue
            if target_label == "Maybe" and _outcome_label(theory, target, hyp_prop) != "Maybe":
                continue

            hyp_atom = Atom(hyp_prop, (target,))
            facts_text, rules_text = render_naf(theory)
            meta = edict(
                facts=facts_text,
                rules=rules_text,
                hypothesis=_capitalized(_naf_atom_text(hyp_atom, "abstract")) + ".",
                conflict_depth=depth,
                target=target,
                label=target_label,
            )
            meta.payload = {"facts": facts_text, "rules": rules_text, "hypothesis": meta.hypothesis}
            label_counts[(depth, target_label)] += 1
            return Entry(meta, target_label)
        raise RuntimeError(f"could not build defeasible conflict depth {depth}")

    def render_prompt(self, meta):
        return (
            "An `unless` condition blocks its default rule.\n\n"
            f"{render_payload(meta.payload)}\n\n"
            "Is the hypothesis true? Answer Yes, No, or Maybe."
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip().lower().rstrip(".") == str(entry.answer).strip().lower())

    def balancing_key(self, problem):
        return (problem.metadata.conflict_depth, problem.answer)


TASK_META = {'parent_source_id': 'c46e36da9bb5649f33aca10cade20290e278794415f21bfd119b16ca23a4c977',
 'idea': 'Test controlled nonmonotonic conflict depth.',
 'hypothesis': 'H1',
 'changes': 'Generate matched default, exception, and exception-to-exception '
            'cases.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 59956088,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 20,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
