import os
import random
from dataclasses import dataclass

import z3

from reasoning_core.template import Config, Entry, Task, edict


ROLES = ('doctor', 'lawyer', 'teacher', 'engineer', 'pilot', 'chef',
         'writer', 'nurse', 'banker', 'farmer', 'scientist', 'baker')
ATTRS = (('tall', 'short'), ('young', 'old'), ('quiet', 'loud'),
         ('kind', 'stern'))
VERBS = ('met', 'called', 'praised', 'questioned', 'greeted', 'thanked')
NAMES = ('John', 'Paul', 'Mark', 'Leo', 'Tom', 'Sam', 'Max', 'Ben',
         'Adam', 'Noah', 'Luke', 'Eric', 'Jack', 'Hugo', 'Alan', 'Owen',
         'Mary', 'Anna', 'Jane', 'Eve', 'Sara', 'Lucy', 'Zoe', 'Rita',
         'Emma', 'Nora', 'Lily', 'Mia', 'Rose', 'Iris', 'Lena', 'Kate')
Z3_TIMEOUT_MS = int(os.environ.get("RC_COREFERENCE_Z3_TIMEOUT_MS", 1000))
# z3's `timeout` is WALL-CLOCK, so a contended machine is indistinguishable from a hard problem.
# Every check then returns `unknown`, every candidate is discarded, generate_entry exhausts
# max_attempts and RAISES -- and the task emits no rows at all rather than degrading. Retrying with
# a larger budget makes starvation cost time instead of data. Set to 0 to restore the old behaviour.
Z3_TIMEOUT_RETRIES = int(os.environ.get("RC_COREFERENCE_Z3_RETRIES", 1))
Z3_TIMEOUT_GROWTH = 5
_z3_unknown_count = 0            # checks that exhausted the whole ladder, for diagnostics


@dataclass(frozen=True)
class Entity:
    eid: int
    name: str
    role: str
    attrs: tuple


@dataclass(frozen=True)
class Factor:
    kind: str
    scope: tuple
    data: tuple = ()
    sentence: int = 0


@dataclass(frozen=True)
class Group:
    variables: tuple
    latent: tuple
    sentence: int


def _solver():
    solver = z3.Solver()
    solver.set(timeout=Z3_TIMEOUT_MS)
    return solver


def _check(solver):
    """`solver.check()` with an escalating wall-clock budget.

    Semantics are unchanged whenever z3 answers within the first budget, which is the normal case --
    these CSPs solve in about a millisecond, so the ladder never escalates on an idle machine. It
    only engages when a check would otherwise return `unknown` and the candidate would be thrown
    away.
    """
    global _z3_unknown_count
    budget = Z3_TIMEOUT_MS
    result = z3.unknown
    for attempt in range(Z3_TIMEOUT_RETRIES + 1):
        if attempt:
            budget *= Z3_TIMEOUT_GROWTH
            solver.set(timeout=budget)
        result = solver.check()
        if result != z3.unknown:
            return result
    _z3_unknown_count += 1
    return result


def _full_description(entity):
    return 'the ' + ' '.join((*entity.attrs, entity.role))


def _introduction(entity):
    words = ' '.join((*entity.attrs, entity.role))
    article = 'an' if words[0].lower() in 'aeiou' else 'a'
    return f"{article} {words} named {entity.name}"


def _join(items):
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ', '.join(items[:-1]) + f", and {items[-1]}"


def _random_permutation(width):
    identity = tuple(range(width))
    for _ in range(100):
        permutation = tuple(random.sample(range(width), width))
        if permutation != identity:
            return permutation
    raise RuntimeError("Could not sample a non-identity permutation")


def compile_z3(domains, factors):
    variables = {idx: z3.Int(f"m{idx}") for idx in domains}
    constraints = [
        z3.Or(*[variables[idx] == value for value in sorted(domain)])
        for idx, domain in domains.items()
    ]
    for factor in factors:
        xs = [variables[idx] for idx in factor.scope]
        if factor.kind == 'equal':
            constraints.append(xs[0] == xs[1])
        elif factor.kind == 'different':
            constraints.append(xs[0] != xs[1])
        elif factor.kind == 'permutation':
            width = len(xs) // 2
            new, old = xs[:width], xs[width:]
            constraints.extend(new[i] == old[source]
                               for i, source in enumerate(factor.data))
        elif factor.kind == 'all_different':
            constraints.append(z3.Distinct(*xs))
        else:
            raise ValueError(factor.kind)
    return variables, constraints


def entailment_status(query_idx, gold, domains, factors):
    variables, constraints = compile_z3(domains, factors)
    solver = _solver()
    solver.add(*constraints)
    result = _check(solver)
    if result == z3.unknown:
        return None
    if result != z3.sat:
        return False
    solver.add(variables[query_idx] != gold)
    result = _check(solver)
    if result == z3.unknown:
        return None
    return result == z3.unsat


def entailed_value(query_idx, domains, factors):
    possible = set()
    variables, constraints = compile_z3(domains, factors)
    for value in domains[query_idx]:
        solver = _solver()
        solver.add(*constraints, variables[query_idx] == value)
        result = _check(solver)
        if result == z3.unknown:
            return None
        if result == z3.sat:
            possible.add(value)
    return next(iter(possible)) if len(possible) == 1 else None


def _all_different_supported(scope, domains, fixed_idx, fixed_value):
    match = {}

    def augment(idx, seen):
        values = ({fixed_value} if idx == fixed_idx else domains[idx])
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            if value not in match or augment(match[value], seen):
                match[value] = idx
                return True
        return False

    ordered = sorted(scope, key=lambda idx: len(domains[idx]))
    return all(augment(idx, set()) for idx in ordered)


def revise_factor(factor, domains):
    revised = {idx: set(domains[idx]) for idx in factor.scope}
    if factor.kind in ('equal', 'different'):
        left, right = factor.scope
        if factor.kind == 'equal':
            shared = domains[left] & domains[right]
            return {left: shared, right: shared}
        revised[left] = {x for x in domains[left]
                         if any(x != y for y in domains[right])}
        revised[right] = {y for y in domains[right]
                          if any(x != y for x in domains[left])}
        return revised
    if factor.kind == 'permutation':
        width = len(factor.scope) // 2
        new, old = factor.scope[:width], factor.scope[width:]
        for i, source in enumerate(factor.data):
            shared = domains[new[i]] & domains[old[source]]
            revised[new[i]] &= shared
            revised[old[source]] &= shared
        return revised
    if factor.kind == 'all_different':
        for idx in factor.scope:
            revised[idx] = {
                value for value in domains[idx]
                if _all_different_supported(factor.scope, domains, idx, value)
            }
        return revised
    raise ValueError(factor.kind)


def propagation_round(domains, factors):
    proposals = {idx: set(values) for idx, values in domains.items()}
    for factor in factors:
        for idx, values in revise_factor(factor, domains).items():
            proposals[idx] &= values
    return proposals


def propagation_trace(query_idx, domains, factors, max_rounds=100):
    current = {idx: set(values) for idx, values in domains.items()}
    trace = [current]
    for _ in range(max_rounds):
        if len(current[query_idx]) == 1:
            return trace
        updated = propagation_round(current, factors)
        if any(not values for values in updated.values()):
            return None
        if updated == current:
            return trace
        current = updated
        trace.append(current)
    raise RuntimeError("Local propagation did not converge")


def propagation_depth(query_idx, domains, factors):
    trace = propagation_trace(query_idx, domains, factors)
    if trace is None or len(trace[-1][query_idx]) != 1:
        return None
    return len(trace) - 1


def necessary_factor_indices(query_idx, gold, domains, factors, indices):
    necessary = []
    for index in indices:
        reduced = factors[:index] + factors[index + 1:]
        status = entailment_status(query_idx, gold, domains, reduced)
        if status is None:
            return None
        if not status:
            necessary.append(index)
    return necessary


@dataclass
class CoreferenceConfig(Config):
    family: str = 'mixed'
    n_entities: int = 4
    target_depth: int = 3
    n_distractor_sentences: int = 2
    max_attempts: int = 200

    def apply_difficulty(self, level):
        self.target_depth += level

    def apply_width_difficulty(self, level):
        self.n_entities += level

    def apply_noise_difficulty(self, level):
        self.n_distractor_sentences += level


class Coreference(Task):
    """Track references through variable-width permutations and constraints."""

    summary = "Resolve references through ordered groups, later evidence, and branches."
    config_cls = CoreferenceConfig

    def _entities(self):
        width = self.config.n_entities
        if not 3 <= width <= 8:
            raise ValueError("n_entities must be between 3 and 8")
        names = random.sample(NAMES, width + 2)
        role = random.choice(ROLES)
        active = [
            Entity(i, names[i], role,
                   tuple(group[(i >> bit) & 1]
                         for bit, group in enumerate(ATTRS)))
            for i in range(width)
        ]
        distractors = [
            Entity(width + i, names[width + i], random.choice(
                [candidate for candidate in ROLES if candidate != role]),
                   ('quiet', 'young'))
            for i in range(2)
        ]
        return active, distractors

    def _sample_family(self):
        family = self.config.family
        allowed = ('permutation_forward', 'permutation_backward',
                   'branching_elimination')
        if family == 'mixed':
            choices = allowed if self.config.target_depth >= 2 else allowed[:2]
            return random.choice(choices)
        if family not in allowed:
            raise ValueError(f"family must be mixed or one of {allowed}")
        if family == 'branching_elimination' and self.config.target_depth < 2:
            raise ValueError("branching_elimination requires target_depth >= 2")
        return family

    def _generate_candidate(self):
        active, distractors = self._entities()
        width = len(active)
        active_ids = {entity.eid for entity in active}
        by_eid = {entity.eid: entity for entity in (*active, *distractors)}
        family = self._sample_family()
        lines, mentions, domains, factors, permutations = [], [], {}, [], []
        next_variable = 0

        def add_line(text):
            sentence = len(lines)
            lines.append(f"({sentence + 1}) {text}")
            return sentence

        def add_group(text, latent, allowed):
            nonlocal next_variable
            sentence = add_line(text)
            variables = tuple(range(next_variable, next_variable + width))
            next_variable += width
            for position, (idx, entity_id, domain) in enumerate(
                    zip(variables, latent, allowed)):
                domains[idx] = set(domain)
                mentions.append({
                    'idx': idx, 'sentence': sentence, 'position': position,
                    'surface': f"position {position + 1}",
                    'entity': by_eid[entity_id],
                })
            return Group(variables, tuple(latent), sentence)

        def add_transition(parent):
            permutation = _random_permutation(width)
            numbers = ', '.join(str(index + 1) for index in permutation)
            latent = tuple(parent.latent[index] for index in permutation)
            child = add_group(
                f"The lineup from sentence {parent.sentence + 1} was reordered "
                f"using positions {numbers}, in that order.",
                latent, [active_ids] * width)
            factors.append(Factor(
                'permutation', child.variables + parent.variables,
                permutation, child.sentence))
            permutations.append({
                'sentence': child.sentence, 'source_sentence': parent.sentence,
                'mapping': tuple(index + 1 for index in permutation),
            })
            return child, len(factors) - 1

        introductions = _join([_introduction(entity) for entity in active])
        support = []

        if family in ('permutation_forward', 'branching_elimination'):
            root = add_group(
                f"{introductions} formed a lineup in that order.",
                [entity.eid for entity in active],
                [{entity.eid} for entity in active])

        if family == 'permutation_forward':
            current = root
            for _ in range(self.config.target_depth):
                current, factor_index = add_transition(current)
                support.append(factor_index)
            query_position = random.randrange(width)
            query_group = current

        elif family == 'permutation_backward':
            add_line(f"{introductions} joined the exercise.")
            role = active[0].role
            root = add_group(
                f"{_join([f'the {role}'] * width).capitalize()} formed a lineup "
                "in that order.",
                random.sample([entity.eid for entity in active], width),
                [active_ids] * width)
            query_position = random.randrange(width)
            query_group = root
            current = root
            for _ in range(self.config.target_depth - 1):
                current, factor_index = add_transition(current)
                support.append(factor_index)
            names = ', '.join(by_eid[eid].name for eid in current.latent)
            anchor = add_group(
                f"The lineup in sentence {current.sentence + 1} was {names}, "
                "in that order.", current.latent,
                [{eid} for eid in current.latent])
            identity = tuple(range(width))
            factors.append(Factor(
                'permutation', anchor.variables + current.variables,
                identity, anchor.sentence))
            support.append(len(factors) - 1)
            permutations.append({
                'sentence': anchor.sentence,
                'source_sentence': current.sentence,
                'mapping': tuple(index + 1 for index in identity),
            })

        else:
            branch_depth = self.config.target_depth - 2
            branches, branch_support = [], []
            for _ in range(2):
                current, path = root, []
                for _ in range(branch_depth):
                    current, factor_index = add_transition(current)
                    path.append(factor_index)
                branches.append(current)
                branch_support.append(path)

            query_position = random.randrange(width)
            merge_latent = tuple(random.sample([entity.eid for entity in active],
                                               width))
            clauses, source_specs = [], []
            specified = [position for position in range(width)
                         if position != query_position]
            random.shuffle(specified)
            for order, output_position in enumerate(specified):
                branch_index = order % 2 if branch_depth else 0
                source = branches[branch_index]
                source_position = source.latent.index(merge_latent[output_position])
                clauses.append(
                    f"position {output_position + 1} came from position "
                    f"{source_position + 1} of sentence {source.sentence + 1}")
                source_specs.append((output_position, source.variables[source_position],
                                     branch_index))
            merge = add_group(
                "A new lineup was formed: " + '; '.join(clauses) + '.',
                merge_latent, [active_ids] * width)
            used_branches = set()
            for output_position, source_variable, branch_index in source_specs:
                factors.append(Factor(
                    'equal', (merge.variables[output_position], source_variable),
                    sentence=merge.sentence))
                support.append(len(factors) - 1)
                used_branches.add(branch_index)
            factors.append(Factor('all_different', merge.variables,
                                  sentence=merge.sentence))
            support.append(len(factors) - 1)
            for branch_index in used_branches:
                support.extend(branch_support[branch_index])
            query_group = merge

        query_idx = query_group.variables[query_position]
        gold = query_group.latent[query_position]
        query_sentence = query_group.sentence

        for _ in range(self.config.n_distractor_sentences):
            first, second = random.sample(distractors, 2)
            add_line(f"Outside the lineup, {first.name} {random.choice(VERBS)} "
                     f"{second.name}.")

        variables, constraints = compile_z3(domains, factors)
        solver = _solver()
        solver.add(*constraints)
        if _check(solver) != z3.sat:      # `unknown` lands here too, and also discards the candidate
            return None
        if entailed_value(query_idx, domains, factors) != gold:
            return None
        trace = propagation_trace(query_idx, domains, factors)
        depth = propagation_depth(query_idx, domains, factors)
        if (trace is None or depth != self.config.target_depth or
                len(trace[-2][query_idx]) <= 1 or
                len(trace[-1][query_idx]) != 1):
            return None
        support = sorted(set(support))
        necessary = necessary_factor_indices(
            query_idx, gold, domains, factors, support)
        if necessary is None or necessary != support:
            return None

        prefix_entailed = None
        if family == 'permutation_backward':
            prefix_factors = [factor for factor in factors
                              if factor.sentence <= query_sentence]
            status = entailment_status(query_idx, gold, domains, prefix_factors)
            if status is None or status:
                return None
            prefix_entailed = False

        audit = lambda state: {str(idx): sorted(values)
                               for idx, values in state.items()}
        metadata = edict({
            'sentences': '\n'.join(lines),
            'family': family,
            'width': width,
            'target_depth': self.config.target_depth,
            'propagation_depth': depth,
            'query_idx': query_idx,
            'query_sentence': query_sentence + 1,
            'query_position': query_position + 1,
            'answer_eid': gold,
            'answer_domain_eids': sorted(active_ids),
            'entities': [*active, *distractors],
            'mentions': mentions,
            'factors': factors,
            'support_factor_indices': support,
            'necessary_factor_indices': necessary,
            'permutations': permutations,
            'initial_domains': audit(domains),
            'propagation_trace': [audit(state) for state in trace],
            'prefix_entailed': prefix_entailed,
            'full_csp_sat': True,
            'z3_entailed': True,
            'realized_distractor_count': self.config.n_distractor_sentences,
        })
        return Entry(metadata=metadata, answer=by_eid[gold].name)

    def generate_entry(self):
        before = _z3_unknown_count
        for _ in range(self.config.max_attempts):
            candidate = self._generate_candidate()
            if candidate is not None:
                return candidate
        # This raise means the task emits NO rows for the job, so say why: a solver-timeout count
        # near the attempt count means z3 is starved (raise RC_COREFERENCE_Z3_TIMEOUT_MS), while
        # zero timeouts means the config itself is infeasible (usually target_depth too high).
        timeouts = _z3_unknown_count - before
        raise RuntimeError(
            f"Could not generate a certified coreference problem in {self.config.max_attempts} "
            f"attempts (z3 timeouts: {timeouts}, target_depth={self.config.target_depth}, "
            f"n_entities={self.config.n_entities}, z3_budget_ms={Z3_TIMEOUT_MS}"
            f"x{Z3_TIMEOUT_RETRIES + 1})")

    def render_prompt(self, metadata):
        return (
            "Each lineup contains the same introduced people exactly once. "
            "A reordering lists the source positions from left to right. "
            "Each description occurrence is independent, and later statements "
            "may resolve earlier ones.\n\n"
            f"{metadata['sentences']}\n\n"
            f"Who occupied position {metadata['query_position']} in sentence "
            f"{metadata['query_sentence']}?\n"
            "The answer is one name."
        )

    def score_answer(self, answer, entry):
        words = str(answer or '').strip().strip('.').strip("'\"").split()
        predicted = (words or [''])[-1].lower()
        return float(predicted == entry.answer.lower())
