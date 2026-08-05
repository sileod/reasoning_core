from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import product

from reasoning_core.template import Entry, Task, render_payload
from reasoning_core.tasks.logic_depth import (
    MultistepNLIConfig,
    _rule_instances,
    atom_text,
    case_metadata,
    derivation_rules,
    generate_case,
    indexed_premise,
)


@dataclass(frozen=True)
class GroundStep:
    atom: object
    rule_line: int
    parents: tuple


@dataclass(frozen=True)
class ProofDag:
    steps: frozenset

    @property
    def size(self):
        return len(self.steps)

    def signature(self):
        return tuple(sorted(
            (
                _atom_key(step.atom),
                step.rule_line,
                tuple(_atom_key(parent) for parent in step.parents),
            )
            for step in self.steps
        ))


@dataclass(frozen=True)
class TraceResult:
    depth: int
    size: int
    traces: tuple
    truncated: bool = False

    @property
    def unique(self):
        return not self.truncated and len(self.traces) == 1

    @property
    def answer(self):
        return self.traces[0] if self.unique else None


def _atom_key(atom):
    return atom.pred, atom.args, atom.sign


def atom_code(atom):
    sign = "" if atom.sign else "!"
    return f"{sign}{atom.pred}({','.join(map(str, atom.args))})"


def _merge_dags(dags, step):
    by_atom = {}
    for dag in dags:
        for existing in dag.steps:
            previous = by_atom.get(existing.atom)
            if previous is not None and previous != existing:
                return None
            by_atom[existing.atom] = existing
    previous = by_atom.get(step.atom)
    if previous is not None and previous != step:
        return None
    by_atom[step.atom] = step
    return ProofDag(frozenset(by_atom.values()))


def _add_candidate(table, atom, depth, dag, max_candidates):
    candidates = table[atom][depth]
    signature = dag.signature()
    if any(candidate.signature() == signature for candidate in candidates):
        return False
    candidates.append(dag)
    candidates.sort(key=lambda candidate: (candidate.size, candidate.signature()))
    truncated = len(candidates) > max_candidates
    del candidates[max_candidates:]
    return truncated


def _ordered_steps(dag, target, facts):
    by_atom = {step.atom: step for step in dag.steps}
    ordered, seen = [], set()

    def visit(atom):
        if atom in facts or atom in seen:
            return
        step = by_atom[atom]
        for parent in step.parents:
            visit(parent)
        seen.add(atom)
        ordered.append(step)

    visit(target)
    return ordered


def render_trace(dag, target, theory, source):
    facts = set(theory.facts)
    steps = _ordered_steps(dag, target, facts)
    step_ids = {step.atom: i for i, step in enumerate(steps)}
    lines = []
    for step in steps:
        supports = [
            str(source[parent]) if parent in facts else f"@{step_ids[parent]}"
            for parent in step.parents
        ]
        lines.append(
            f"{step.rule_line}: {' '.join(supports) or '-'} => {atom_code(step.atom)}"
        )
    return "\n".join(lines)


def canonical_trace(theory, res, source, target, max_depth=None, max_candidates=128):
    """Return canonical minimum-depth, minimum-step proof traces for ``target``.

    Candidates are proof DAGs, so a derived fact reused by several later steps is
    emitted once. If the bounded search truncates, the result is marked ambiguous
    and generation rejects it rather than claiming a false unique answer.
    """
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2")
    if target not in res.derivations:
        return None

    target_depth = res.derivations[target].depth
    max_depth = target_depth if max_depth is None else min(int(max_depth), target_depth)
    if max_depth < target_depth:
        return None

    facts = set(theory.facts)
    table = defaultdict(lambda: defaultdict(list))
    for fact in theory.facts:
        if fact in source:
            table[fact][0].append(ProofDag(frozenset()))

    instances = [
        (source[rule], head, parents)
        for rule in theory.rules
        if rule in source
        for head, parents in _rule_instances(rule, res.closure)
        if head not in facts
    ]

    truncated = False
    for depth in range(1, max_depth + 1):
        for rule_line, head, parents in instances:
            choices = []
            for parent in parents:
                options = [
                    (parent_depth, dag)
                    for parent_depth, dags in table[parent].items()
                    if parent_depth < depth
                    for dag in dags
                ]
                if not options:
                    break
                choices.append(options)
            else:
                combinations = product(*choices) if choices else [()]
                for combination in combinations:
                    parent_depths = [item[0] for item in combination]
                    if 1 + max(parent_depths, default=0) != depth:
                        continue
                    step = GroundStep(
                        head,
                        rule_line,
                        tuple(parents),
                    )
                    dag = _merge_dags([item[1] for item in combination], step)
                    if dag is not None:
                        truncated |= _add_candidate(
                            table, head, depth, dag, max_candidates
                        )

    candidates = table[target].get(target_depth, ())
    if not candidates:
        return None
    size = min(candidate.size for candidate in candidates)
    traces = tuple(sorted({
        render_trace(candidate, target, theory, source)
        for candidate in candidates
        if candidate.size == size
    }))
    return TraceResult(target_depth, size, traces, truncated)


def normalize_trace(text):
    return "\n".join(
        "".join(line.split())
        for line in str(text).strip().splitlines()
        if line.strip()
    )


class LogicDerivation(Task):
    summary = "Produce a canonical forward proof trace for a logical target."

    def __init__(self, config=None):
        super().__init__(config=config or MultistepNLIConfig())
        self.balancing_key_ratio = 1 / 3
        self._case_state = {}

    def generate_entry(self):
        for _ in range(300):
            trial_state = deepcopy(self._case_state)
            case, key = generate_case(
                self.config,
                ("entailment", "contradiction"),
                trial_state,
            )
            if not case:
                continue

            trace = canonical_trace(
                case.theory,
                case.res,
                case.source,
                case.target,
                max_depth=case.derivation.depth,
            )
            if trace is None or not trace.unique:
                continue

            self._case_state = trial_state
            rules = derivation_rules(case.target, case.res.derivations)
            meta = case_metadata(case, key)
            meta.target = atom_text(case.target, case.theory.domain_pack) + "."
            meta.proof_depth = trace.depth
            meta.proof_steps = trace.size
            meta.optimal_trace_count = len(trace.traces)
            meta.uses_binary = any(
                len(atom.args) == 2 for atom in case.support_atoms | {case.target}
            )
            meta.uses_signed = any(not rule.head.sign for rule in rules)
            meta.uses_composition = any("composition" in rule.shape for rule in rules)
            meta.payload = {
                "premise": indexed_premise(case.lines),
                "target": meta.target,
            }
            return Entry(meta, trace.answer)

        raise RuntimeError("could not generate a unique logic_derivation example")

    def render_prompt(self, meta):
        return (
            f"{render_payload(meta.payload)}\n\n"
            "Give the proof, one step per line: `rule: supports => conclusion`. "
            "Use premise lines for facts, `@i` for earlier proof lines, and keep "
            "supports in rule-condition order. Write conclusions as `p(x)` or `!p(x)`."
        )

    def score_answer(self, answer, entry):
        return float(normalize_trace(answer) == normalize_trace(entry.answer))

    def balancing_key(self, problem):
        meta = problem.metadata
        return (
            meta.domain_pack,
            min(meta.proof_depth, 4),
            min(meta.proof_steps, 6),
            meta.uses_binary,
            meta.uses_signed,
            meta.uses_composition,
        )
