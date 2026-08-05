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
class Proof:
    line: int
    children: tuple = None

    @property
    def size(self):
        return 1 + sum(child.size for child in self.children or ())

    def render(self):
        if self.children is None:
            return str(self.line)
        return f"{self.line}({','.join(child.render() for child in self.children)})"


@dataclass(frozen=True)
class ProofResult:
    depth: int
    size: int
    proofs: tuple

    @property
    def unique(self):
        return len(self.proofs) == 1

    @property
    def answer(self):
        return self.proofs[0].render() if self.unique else None


def _add_proof(table, atom, depth, proof, limit):
    size = proof.size
    current = table[atom].get(depth)
    if current is None or size < current[0]:
        table[atom][depth] = size, (proof,)
        return
    if size != current[0]:
        return
    proofs = {p.render(): p for p in (*current[1], proof)}
    table[atom][depth] = size, tuple(proofs[k] for k in sorted(proofs)[:limit])


def canonical_proof(theory, res, source, target, max_depth=None, limit=2):
    """Return up to ``limit`` optimal proof terms for ``target``.

    Proofs are ordered by derivation depth, then by proof-tree size. Keeping two
    terms is sufficient to distinguish a unique optimum from an ambiguous one.
    """
    if limit < 2:
        raise ValueError("limit must be at least 2 to detect ambiguity")
    if target not in res.derivations:
        return None

    target_depth = res.derivations[target].depth
    max_depth = target_depth if max_depth is None else min(int(max_depth), target_depth)
    if max_depth < target_depth:
        return None

    table = defaultdict(dict)
    for fact in theory.facts:
        if fact in source:
            _add_proof(table, fact, 0, Proof(source[fact]), limit)

    instances = [
        (source[rule], head, parents)
        for rule in theory.rules
        if rule in source
        for head, parents in _rule_instances(rule, res.closure)
    ]

    for depth in range(1, max_depth + 1):
        for line, head, parents in instances:
            if not parents:
                if depth == 1:
                    _add_proof(table, head, depth, Proof(line, ()), limit)
                continue

            choices = []
            for parent in parents:
                options = [
                    (parent_depth, proof)
                    for parent_depth, (_, proofs) in table[parent].items()
                    if parent_depth < depth
                    for proof in proofs
                ]
                if not options:
                    break
                choices.append(options)
            else:
                for combination in product(*choices):
                    if 1 + max(parent_depth for parent_depth, _ in combination) != depth:
                        continue
                    proof = Proof(line, tuple(proof for _, proof in combination))
                    _add_proof(table, head, depth, proof, limit)

    current = table[target].get(target_depth)
    if current is None:
        return None
    size, proofs = current
    return ProofResult(target_depth, size, proofs)


class LogicDerivation(Task):
    summary = "Produce a canonical proof certificate for a logical target."

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

            proof = canonical_proof(
                case.theory,
                case.res,
                case.source,
                case.target,
                max_depth=case.derivation.depth,
            )
            if proof is None or not proof.unique:
                continue

            self._case_state = trial_state
            rules = derivation_rules(case.target, case.res.derivations)
            meta = case_metadata(case, key)
            meta.target = atom_text(case.target, case.theory.domain_pack) + "."
            meta.proof_depth = proof.depth
            meta.proof_size = proof.size
            meta.optimal_proof_count = len(proof.proofs)
            meta.uses_binary = any(
                len(atom.args) == 2 for atom in case.support_atoms | {case.target}
            )
            meta.uses_signed = any(not rule.head.sign for rule in rules)
            meta.uses_composition = any("composition" in rule.shape for rule in rules)
            meta.payload = {
                "premise": indexed_premise(case.lines),
                "target": meta.target,
            }
            return Entry(meta, proof.answer)

        raise RuntimeError("could not generate a unique logic_derivation example")

    def render_prompt(self, meta):
        return (
            f"{render_payload(meta.payload)}\n\n"
            "Give the unique optimal proof of the target. A fact on line [2] is "
            "written `2`; applying rule [7] to proofs `2` and `4` is written "
            "`7(2,4)`. Arguments follow the rule conditions from left to right. "
            "Optimal means minimum derivation depth, then the fewest proof nodes. "
            "Answer only with the proof."
        )

    def score_answer(self, answer, entry):
        normalize = lambda x: "".join(str(x).split())
        return float(normalize(answer) == normalize(entry.answer))

    def balancing_key(self, problem):
        meta = problem.metadata
        return (
            meta.domain_pack,
            min(meta.proof_depth, 4),
            meta.uses_binary,
            meta.uses_signed,
            meta.uses_composition,
        )
