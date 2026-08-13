"""Select a symbolic expression compiled from a latent counting program."""

from __future__ import annotations

import ast
import math
import random
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from itertools import product

from reasoning_core.template import Config, Entry, Task, edict, render_payload


@dataclass
class CombinatoricsConfig(Config):
    depth_2_rate: float = 0.2
    depth_3_rate: float = 0.05
    explicit_rate: float = 0.9
    multiple_choice_prob: float = 0.2
    max_tries: int = 100

    def apply_difficulty(self, level):
        self.depth_2_rate = min(0.65, self.depth_2_rate + level * 0.09)
        self.depth_3_rate = min(0.55, self.depth_3_rate + level * 0.1)
        self.explicit_rate = max(0.25, self.explicit_rate - level * 0.13)


@dataclass(frozen=True)
class Select:
    population: int
    size: int
    ordered: bool
    replacement: bool


@dataclass(frozen=True)
class Distribute:
    objects: int
    boxes: int
    require_nonempty: bool


@dataclass(frozen=True)
class Arrange:
    objects: int
    circular: bool = False
    adjacent_pair: bool = False


@dataclass(frozen=True)
class UnionCount:
    first: int
    second: int
    overlap: int


@dataclass(frozen=True)
class ChoiceRule:
    kind: str
    first: int
    second: int


@dataclass(frozen=True)
class LatticePath:
    right: int
    up: int


@dataclass(frozen=True)
class RoleThenCommittee:
    population: int
    committee_size: int


@dataclass(frozen=True)
class ExactSymbolStrings:
    length: int
    alphabet: int
    exact: int


@dataclass(frozen=True)
class ManagerCommittee:
    population: int
    managers: int
    size: int


@dataclass(frozen=True)
class ThroughPointPath:
    right: int
    up: int
    point_right: int
    point_up: int


@dataclass(frozen=True)
class ExclusiveCommittee:
    population: int
    size: int


@dataclass(frozen=True)
class ProjectChairCommittee:
    projects: int
    population: int
    committee_size: int


@dataclass(frozen=True)
class TwoCheckpointPath:
    right: int
    up: int
    first_right: int
    first_up: int
    second_right: int
    second_up: int


@dataclass(frozen=True)
class Option:
    expression: str
    value: int
    mutation: str
    correct: bool = False
    semantics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledProblem:
    family: str
    depth: int
    text: str
    options: tuple[Option, ...]


def _option(expression, value, mutation, correct=False, **semantics):
    return Option(expression, value, mutation, correct, semantics)


def _surface(explicit, explicit_forms, implicit_forms):
    return random.choice(explicit_forms if explicit else implicit_forms)


def _selection_expression(n, k, ordered, replacement):
    if ordered and replacement:
        return f"{n}^{k}", n**k
    if ordered:
        return f"P({n},{k})", math.perm(n, k)
    if replacement:
        return f"C({n + k - 1},{k})", math.comb(n + k - 1, k)
    return f"C({n},{k})", math.comb(n, k)


def _selection_text(program, explicit):
    n, k = program.population, program.size
    forms = {
        (False, False): (
            (
                f"Choose {k} different items from {n}; order does not matter.",
                f"Pick {k} people from {n}, without repetition and without order.",
            ),
            (
                f"Select a {k}-member subset from {n} distinct items.",
                f"Form a committee of {k} from {n} people.",
                f"Form a group of {k} from {n} distinct people.",
            ),
        ),
        (True, False): (
            (
                f"Choose {k} different items from {n} in order.",
                f"Select an ordered list of {k} distinct objects from {n}.",
            ),
            (
                f"Assign {k} distinct roles among {n} people.",
                f"Award first through {k}th place among {n} runners.",
                f"Choose a president, then {k - 1} ranked officers, from {n} people.",
            ),
        ),
        (True, True): (
            (
                f"Form a length-{k} sequence from {n} symbols; repetition is allowed.",
                f"Make {k} ordered choices from {n} options, allowing repeats.",
            ),
            (
                f"Create a length-{k} code from an alphabet of {n} symbols.",
                f"Record {k} rolls of an {n}-sided die.",
                f"Fill {k} ordered slots, each with one of {n} symbols.",
            ),
        ),
        (False, True): (
            (
                f"Choose {k} items from {n} types; repetition is allowed and order is irrelevant.",
                f"Select {k} objects from {n} kinds, with repetition but without order.",
            ),
            (
                f"Choose {k} scoops from {n} flavors; flavors may repeat.",
                f"Select a multiset of size {k} from {n} item types.",
                f"Buy {k} pieces of fruit chosen from {n} varieties, with varieties reusable.",
            ),
        ),
    }
    explicit_forms, implicit_forms = forms[(program.ordered, program.replacement)]
    return _surface(explicit, explicit_forms, implicit_forms)


def _compile_select(program, explicit):
    n, k = program.population, program.size
    options = []
    for ordered, replacement in ((False, False), (True, False), (True, True), (False, True)):
        correct = (ordered, replacement) == (program.ordered, program.replacement)
        changes = []
        if ordered != program.ordered:
            changes.append("adds_order" if ordered else "drops_order")
        if replacement != program.replacement:
            changes.append("allows_repetition" if replacement else "forbids_repetition")
        expression, value = _selection_expression(n, k, ordered, replacement)
        options.append(
            _option(
                expression,
                value,
                "correct" if correct else "+".join(changes),
                correct,
                ordered=ordered,
                replacement=replacement,
            )
        )
    return CompiledProblem("selection", 1, _selection_text(program, explicit), tuple(options))


def _compile_distribute(program, explicit):
    n, b = program.objects, program.boxes
    if program.require_nonempty:
        text = _surface(
            explicit,
            (
                f"Place {n} identical tokens into {b} labeled boxes, with every box nonempty.",
                f"Distribute {n} identical objects among {b} labeled containers; none may be empty.",
            ),
            (
                f"Split {n} identical candies among {b} named children, giving each child at least one.",
                f"Write {n} as a sum of {b} positive labeled parts.",
            ),
        )
        options = (
            _option(f"C({n - 1},{b - 1})", math.comb(n - 1, b - 1), "correct", True,
                    objects_labeled=False, boxes_labeled=True, require_nonempty=True, bar_offset=0),
            _option(f"C({n + b - 1},{b - 1})", math.comb(n + b - 1, b - 1), "allows_empty_boxes",
                    objects_labeled=False, boxes_labeled=True, require_nonempty=False, bar_offset=0),
            _option(f"C({n - 1},{b})", math.comb(n - 1, b), "uses_one_extra_bar",
                    objects_labeled=False, boxes_labeled=True, require_nonempty=True, bar_offset=1),
            _option(f"{b}^{n}", b**n, "treats_objects_as_distinct",
                    objects_labeled=True, boxes_labeled=True, require_nonempty=False),
        )
    else:
        text = _surface(
            explicit,
            (
                f"Place {n} identical tokens into {b} labeled boxes; empty boxes are allowed.",
                f"Distribute {n} identical objects among {b} labeled containers, allowing empties.",
            ),
            (
                f"Split {n} identical candies among {b} named children; a child may receive none.",
                f"Write {n} as a sum of {b} nonnegative labeled parts.",
            ),
        )
        options = (
            _option(f"C({n + b - 1},{b - 1})", math.comb(n + b - 1, b - 1), "correct", True,
                    objects_labeled=False, boxes_labeled=True, require_nonempty=False, star_offset=0),
            _option(f"C({n - 1},{b - 1})", math.comb(n - 1, b - 1), "requires_nonempty_boxes",
                    objects_labeled=False, boxes_labeled=True, require_nonempty=True, star_offset=0),
            _option(f"C({n + b},{b - 1})", math.comb(n + b, b - 1), "uses_one_extra_star",
                    objects_labeled=False, boxes_labeled=True, require_nonempty=False, star_offset=1),
            _option(f"{b}^{n}", b**n, "treats_objects_as_distinct",
                    objects_labeled=True, boxes_labeled=True, require_nonempty=False),
        )
    return CompiledProblem("distribution", 1, text, options)


def _compile_arrange(program, explicit):
    n = program.objects
    if program.circular and program.adjacent_pair:
        raise ValueError("circular adjacency is not supported")
    if program.adjacent_pair:
        text = _surface(
            explicit,
            (
                f"Arrange {n} distinct books in a row, with two specified books adjacent.",
                f"Order {n} distinct objects linearly; a specified pair must stay together.",
            ),
            (
                f"Line up {n} people so that Alice and Bob stand next to each other.",
                f"Shelve {n} different books with two named volumes side by side.",
            ),
        )
        options = (
            _option(f"2*{n - 1}!", 2 * math.factorial(n - 1), "correct", True,
                    block_units=n - 1, internal_orders=2),
            _option(f"{n - 1}!", math.factorial(n - 1), "forgets_internal_order",
                    block_units=n - 1, internal_orders=1),
            _option(f"2*{n - 2}!", 2 * math.factorial(n - 2), "wrong_number_of_units",
                    block_units=n - 2, internal_orders=2),
            _option(f"{n}!-2*{n - 1}!", math.factorial(n) - 2 * math.factorial(n - 1),
                    "takes_complement", direct=False),
        )
        return CompiledProblem("block_arrangement", 1, text, options)

    if program.circular:
        text = _surface(
            explicit,
            (
                f"Arrange {n} distinct people in a circle, identifying rotations but not reflections.",
                f"Seat {n} distinct people around a circular table; rotations are the same.",
            ),
            (
                f"Seat {n} distinct guests at a round table with no distinguished seat.",
                f"Make a necklace ordering of {n} distinct labeled beads, keeping mirror images different.",
            ),
        )
    else:
        text = _surface(
            explicit,
            (f"Arrange {n} distinct people in a row.", f"Order {n} distinct objects linearly."),
            (f"Line up {n} different people.", f"Place {n} different books on a shelf."),
        )
    options = (
        _option(f"{n}!", math.factorial(n), "correct" if not program.circular else "ignores_rotations",
                not program.circular, circular=False, reflection_equivalent=False),
        _option(f"{n - 1}!", math.factorial(n - 1), "correct" if program.circular else "identifies_rotations",
                program.circular, circular=True, reflection_equivalent=False),
        _option(f"{n}!/2", math.factorial(n) // 2, "confuses_rotation_with_reflection",
                circular=False, reflection_equivalent=True),
        _option(f"{n - 1}!/2", math.factorial(n - 1) // 2, "also_identifies_reflections",
                circular=True, reflection_equivalent=True),
    )
    family = "circular_arrangement" if program.circular else "linear_arrangement"
    return CompiledProblem(family, 1, text, options)


def _compile_union(program, explicit):
    a, b, c = program.first, program.second, program.overlap
    text = _surface(
        explicit,
        (
            f"Set A has {a} elements, set B has {b}, and their intersection has {c}. Count their union.",
            f"Of some people, {a} satisfy A, {b} satisfy B, and {c} satisfy both. Count those satisfying A or B.",
        ),
        (
            f"In a survey, {a} people study French, {b} study Spanish, and {c} study both. Count those studying at least one.",
            f"A club has {a} chess players and {b} tennis players; {c} play both. Count players of either game.",
        ),
    )
    options = (
        _option(f"{a}+{b}-{c}", a + b - c, "correct", True, intersection_copies_subtracted=1),
        _option(f"{a}+{b}", a + b, "omits_intersection", intersection_copies_subtracted=0),
        _option(f"{a}+{b}+{c}", a + b + c, "adds_intersection", intersection_copies_subtracted=-1),
        _option(f"{a}+{b}-2*{c}", a + b - 2 * c, "subtracts_intersection_twice",
                intersection_copies_subtracted=2),
    )
    return CompiledProblem("inclusion_exclusion", 1, text, options)


def _compile_choice(program, explicit):
    a, b = program.first, program.second
    if program.kind == "product":
        text = _surface(
            explicit,
            (f"Choose one item from each of two labeled groups of sizes {a} and {b}.",
             f"Make a first choice in {a} ways and an independent second choice in {b} ways."),
            (f"Choose one of {a} shirts and one of {b} pairs of trousers.",
             f"A meal has one of {a} mains and one of {b} desserts."),
        )
        correct_expression = f"{a}*{b}"
    elif program.kind == "sum":
        text = _surface(
            explicit,
            (f"Choose exactly one item from disjoint groups of sizes {a} and {b}.",
             f"Make one choice of type A in {a} ways or type B in {b} ways; the types do not overlap."),
            (f"Choose one elective: one of {a} art courses or one of {b} science courses.",
             f"Order one dessert: one of {a} cakes or one of {b} pies."),
        )
        correct_expression = f"{a}+{b}"
    elif program.kind == "complement":
        text = _surface(
            explicit,
            (f"Count length-{b} strings over {a} symbols that contain at least one specified symbol.",
             f"Make {b} choices from {a} symbols with repetition, excluding sequences that avoid a fixed symbol."),
            (f"How many length-{b} codes over {a} symbols use x at least once?",
             f"Roll an {a}-sided die {b} times and require at least one roll of 1."),
        )
        options = (
            _option(f"{a}^{b}-{a - 1}^{b}", a**b - (a - 1) ** b, "correct", True,
                    uses_complement=True, forbidden_symbols=a - 1),
            _option(f"{a}^{b}-1", a**b - 1, "removes_one_outcome",
                    uses_complement=True, forbidden_outcomes=1),
            _option(f"{a - 1}^{b}", (a - 1) ** b, "counts_complement",
                    uses_complement=False, avoids_symbol=True),
            _option(f"{b}*{a - 1}^{b - 1}", b * (a - 1) ** (b - 1), "requires_exactly_one",
                    exact_occurrences=1),
        )
        return CompiledProblem("complement", 1, text, options)
    else:
        raise ValueError(f"unknown choice rule: {program.kind}")

    alternatives = (
        (f"{a}+{b}", a + b, "uses_sum_rule", "sum"),
        (f"{a}*{b}", a * b, "uses_product_rule", "product"),
        (f"{a}+{b}-1", a + b - 1, "subtracts_one_outcome", "adjusted_sum"),
        (f"2*{a}*{b}", 2 * a * b, "counts_two_orders", "ordered_product"),
    )
    options = tuple(
        _option(expression, value, "correct" if expression == correct_expression else mutation,
                expression == correct_expression, rule=rule)
        for expression, value, mutation, rule in alternatives
    )
    return CompiledProblem(f"{program.kind}_rule", 1, text, options)


def _compile_path(program, explicit):
    r, u, total = program.right, program.up, program.right + program.up
    text = _surface(
        explicit,
        (f"Count paths from (0,0) to ({r},{u}) using only unit right and up steps.",
         f"A path has {r} right steps and {u} up steps. Count the possible step orders."),
        (f"On a square grid, walk from (0,0) to ({r},{u}) moving only east or north. Count the routes.",
         f"Interleave {r} symbols R and {u} symbols U."),
    )
    options = (
        _option(f"C({total},{r})", math.comb(total, r), "correct", True,
                total_steps=total, chosen_right_steps=r),
        _option(f"C({total},{r - 1})", math.comb(total, r - 1), "uses_one_fewer_right_step",
                total_steps=total, chosen_right_steps=r - 1),
        _option(f"C({total - 1},{r})", math.comb(total - 1, r), "omits_one_step_position",
                total_steps=total - 1, chosen_right_steps=r),
        _option(f"C({total + 1},{r})", math.comb(total + 1, r), "adds_one_step_position",
                total_steps=total + 1, chosen_right_steps=r),
    )
    return CompiledProblem("lattice_path", 1, text, options)


def _compile_role_committee(program, explicit):
    n, k = program.population, program.committee_size
    text = _surface(
        explicit,
        (f"From {n} people, choose a president, then a {k}-person committee excluding the president.",
         f"Assign one leader from {n} people and choose {k} different remaining people as an unordered team."),
        (f"A group of {n} chooses a chair and a separate {k}-member committee.",
         f"Choose a captain, then {k} other people from a squad of {n}."),
    )
    options = (
        _option(f"{n}*C({n - 1},{k})", n * math.comb(n - 1, k), "correct", True,
                stages=2, leader_ordered=True, leader_excluded=True, committee_ordered=False),
        _option(f"{n}*C({n},{k})", n * math.comb(n, k), "allows_leader_on_committee",
                stages=2, leader_ordered=True, leader_excluded=False, committee_ordered=False),
        _option(f"{n}*P({n - 1},{k})", n * math.perm(n - 1, k), "orders_committee",
                stages=2, leader_ordered=True, leader_excluded=True, committee_ordered=True),
        _option(f"{n}*C({n - 1},{k - 1})", n * math.comb(n - 1, k - 1), "counts_leader_in_team_size",
                stages=2, leader_excluded=True, committee_size=k - 1),
    )
    return CompiledProblem("role_then_committee", 2, text, options)


def _compile_exact_strings(program, explicit):
    length, alphabet, exact = program.length, program.alphabet, program.exact
    other, remaining = alphabet - 1, length - exact
    text = _surface(
        explicit,
        (f"Count length-{length} strings over {alphabet} symbols with exactly {exact} copies of x.",
         f"In a sequence of length {length} over {alphabet} symbols, x must occur exactly {exact} times."),
        (f"Fill {length} slots from an alphabet of {alphabet}; exactly {exact} slots contain x.",
         f"How many {length}-character codes over {alphabet} symbols use x exactly {exact} times?"),
    )
    options = (
        _option(f"C({length},{exact})*{other}^{remaining}", math.comb(length, exact) * other**remaining,
                "correct", True, choose_x_positions=exact, fill_positions=remaining, fill_symbols=other),
        _option(f"C({length},{exact})*{alphabet}^{remaining}", math.comb(length, exact) * alphabet**remaining,
                "allows_x_in_other_positions", choose_x_positions=exact, fill_positions=remaining,
                fill_symbols=alphabet),
        _option(f"P({length},{exact})*{other}^{remaining}", math.perm(length, exact) * other**remaining,
                "orders_x_positions", ordered_positions=True, fill_symbols=other),
        _option(f"C({length},{exact})*{other}^{exact}", math.comb(length, exact) * other**exact,
                "fills_wrong_number_of_positions", choose_x_positions=exact, fill_positions=exact,
                fill_symbols=other),
    )
    return CompiledProblem("exact_symbol_strings", 2, text, options)


def _compile_manager_committee(program, explicit):
    n, m, k = program.population, program.managers, program.size
    text = _surface(
        explicit,
        (f"Choose {k} people from {n}, including at least one of {m} managers.",
         f"Count {k}-person subsets of {n} people that contain one or more of the {m} managers."),
        (f"A {k}-member committee is chosen from {n} staff, of whom {m} are managers; a manager must serve.",
         f"Form a team of {k} from {n} people, not entirely from the {n - m} nonmanagers."),
    )
    options = (
        _option(f"C({n},{k})-C({n - m},{k})", math.comb(n, k) - math.comb(n - m, k), "correct", True,
                uses_complement=True, excluded_pool=n - m, excluded_size=k),
        _option(f"C({n},{k})-C({n - m},{k - 1})", math.comb(n, k) - math.comb(n - m, k - 1),
                "wrong_excluded_committee_size", uses_complement=True, excluded_pool=n - m,
                excluded_size=k - 1),
        _option(f"C({n - m},{k})", math.comb(n - m, k), "counts_no_manager_committees",
                uses_complement=False, manager_count=0),
        _option(f"C({n},{k})-C({n - m + 1},{k})", math.comb(n, k) - math.comb(n - m + 1, k),
                "treats_one_manager_as_nonmanager", uses_complement=True, excluded_pool=n - m + 1,
                excluded_size=k),
    )
    return CompiledProblem("manager_committee", 2, text, options)


def _compile_through_point(program, explicit):
    r, u, pr, pu = program.right, program.up, program.point_right, program.point_up
    first_total, second_total = pr + pu, r + u - pr - pu
    second_right = r - pr
    text = _surface(
        explicit,
        (f"Count right/up paths from (0,0) to ({r},{u}) that pass through ({pr},{pu}).",
         f"A north-east lattice path ends at ({r},{u}) and must visit ({pr},{pu}). Count the paths."),
        (f"Walk east or north from (0,0) to ({r},{u}), stopping at ({pr},{pu}) on the way. Count the routes.",
         f"Interleave right and up steps to ({r},{u}), with the prefix ending at ({pr},{pu})."),
    )
    options = (
        _option(f"C({first_total},{pr})*C({second_total},{second_right})",
                math.comb(first_total, pr) * math.comb(second_total, second_right), "correct", True,
                split_at_point=True, first_right=pr, second_right=second_right),
        _option(f"C({r + u},{r})", math.comb(r + u, r), "ignores_required_point",
                split_at_point=False),
        _option(f"C({first_total},{pr})*C({second_total - 1},{second_right})",
                math.comb(first_total, pr) * math.comb(second_total - 1, second_right),
                "omits_step_after_point", split_at_point=True, first_right=pr,
                second_steps=second_total - 1),
        _option(f"C({first_total - 1},{pr})*C({second_total},{second_right})",
                math.comb(first_total - 1, pr) * math.comb(second_total, second_right),
                "omits_step_before_point", split_at_point=True, first_steps=first_total - 1,
                second_right=second_right),
    )
    return CompiledProblem("path_through_point", 2, text, options)


def _compile_exclusive_committee(program, explicit):
    n, k = program.population, program.size
    text = _surface(
        explicit,
        (f"Choose {k} people from {n}, including exactly one of two specified people.",
         f"Count {k}-person subsets of {n} that contain one, but not both, of Alice and Bob."),
        (f"A {k}-member committee from {n} must include exactly one of its two captains.",
         f"Form a group of {k} from {n}; Alice or Bob must join, but they cannot both join."),
    )
    term = f"C({n - 2},{k - 1})"
    options = (
        _option(f"{term}+{term}", 2 * math.comb(n - 2, k - 1), "correct", True,
                disjoint_cases=2, selected_special=1),
        _option(term, math.comb(n - 2, k - 1), "omits_one_case",
                disjoint_cases=1, selected_special=1),
        _option(f"{term}+C({n - 2},{k - 2})", math.comb(n - 2, k - 1) + math.comb(n - 2, k - 2),
                "second_case_includes_both", disjoint_cases=2, second_selected_special=2),
        _option(f"C({n},{k})-C({n - 2},{k})", math.comb(n, k) - math.comb(n - 2, k),
                "requires_at_least_one", exact=False, minimum_special=1),
    )
    return CompiledProblem("exclusive_committee", 2, text, options)


def _compile_project_chair_committee(program, explicit):
    g, n, k = program.projects, program.population, program.committee_size
    text = _surface(
        explicit,
        (f"Choose one of {g} projects, appoint its chair from {n} people, then choose an unordered "
         f"{k}-person committee from the remaining people.",),
        (f"A club picks one of {g} projects, names one of {n} members to lead it, and assigns "
         f"{k} other members as an unordered team.",),
    )
    options = (
        _option(f"{g}*{n}*C({n - 1},{k})", g * n * math.comb(n - 1, k), "correct", True,
                chooses_project=True, chair_excluded=True, committee_ordered=False),
        _option(f"{n}*C({n - 1},{k})", n * math.comb(n - 1, k), "forgets_project_choice",
                chooses_project=False, chair_excluded=True, committee_ordered=False),
        _option(f"{g}*{n}*C({n},{k})", g * n * math.comb(n, k), "allows_chair_on_committee",
                chooses_project=True, chair_excluded=False, committee_ordered=False),
        _option(f"{g}*{n}*P({n - 1},{k})", g * n * math.perm(n - 1, k), "orders_committee",
                chooses_project=True, chair_excluded=True, committee_ordered=True),
    )
    return CompiledProblem("project_chair_committee", 3, text, options)


def _compile_two_checkpoint_path(program, explicit):
    r, u = program.right, program.up
    ar, au = program.first_right, program.first_up
    br, bu = program.second_right, program.second_up
    first = (ar + au, ar)
    middle = (br - ar + bu - au, br - ar)
    last = (r - br + u - bu, r - br)

    def choose(segment):
        return math.comb(*segment)

    text = _surface(
        explicit,
        (f"Count right/up paths from (0,0) to ({r},{u}) that pass first through ({ar},{au}) "
         f"and later through ({br},{bu}).",),
        (f"Walk east or north from (0,0) to ({r},{u}), visiting ({ar},{au}) and then "
         f"({br},{bu}). Count the routes.",),
    )
    direct_second = (br + bu, br)
    after_first = (r - ar + u - au, r - ar)
    correct = choose(first) * choose(middle) * choose(last)
    options = (
        _option(f"C({first[0]},{first[1]})*C({middle[0]},{middle[1]})*C({last[0]},{last[1]})",
                correct, "correct", True, uses_first=True, uses_second=True, segments=3),
        _option(f"C({direct_second[0]},{direct_second[1]})*C({last[0]},{last[1]})",
                choose(direct_second) * choose(last), "ignores_first_checkpoint",
                uses_first=False, uses_second=True, segments=2),
        _option(f"C({first[0]},{first[1]})*C({after_first[0]},{after_first[1]})",
                choose(first) * choose(after_first), "ignores_second_checkpoint",
                uses_first=True, uses_second=False, segments=2),
        _option(f"C({first[0]},{first[1]})*C({direct_second[0]},{direct_second[1]})*C({last[0]},{last[1]})",
                choose(first) * choose(direct_second) * choose(last), "restarts_middle_from_origin",
                uses_first=True, uses_second=True, middle_origin="origin"),
    )
    return CompiledProblem("two_checkpoint_path", 3, text, options)


def _compile(program, explicit=True):
    compilers = {
        Select: _compile_select,
        Distribute: _compile_distribute,
        Arrange: _compile_arrange,
        UnionCount: _compile_union,
        ChoiceRule: _compile_choice,
        LatticePath: _compile_path,
        RoleThenCommittee: _compile_role_committee,
        ExactSymbolStrings: _compile_exact_strings,
        ManagerCommittee: _compile_manager_committee,
        ThroughPointPath: _compile_through_point,
        ExclusiveCommittee: _compile_exclusive_committee,
        ProjectChairCommittee: _compile_project_chair_committee,
        TwoCheckpointPath: _compile_two_checkpoint_path,
    }
    compiler = compilers.get(type(program))
    if compiler is None:
        raise TypeError(f"unsupported counting program: {type(program).__name__}")
    return compiler(program, explicit)


def _valid(compiled):
    return (
        len(compiled.options) == 4
        and sum(option.correct for option in compiled.options) == 1
        and len({option.expression for option in compiled.options}) == 4
        and len({option.value for option in compiled.options}) == 4
        and all(option.value > 0 for option in compiled.options)
        and len({option.mutation for option in compiled.options}) == 4
        and all(option.semantics for option in compiled.options)
    )


def _evaluate_expression(expression):
    source = re.sub(r"(\d+)!", r"F(\1)", expression).replace("^", "**")

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (right.denominator != 1 or abs(right) > 100):
                raise ValueError("exponent outside finite grammar bound")
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.Pow: lambda: left ** int(right),
            }
            if type(node.op) in operations:
                value = operations[type(node.op)]()
                if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
                    raise ValueError("counting value outside finite grammar bound")
                return value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [int(visit(arg)) for arg in node.args]
            functions = {"C": math.comb, "P": math.perm, "F": math.factorial}
            if node.func.id in functions:
                return Fraction(functions[node.func.id](*args))
        raise ValueError("unsupported counting expression")

    return visit(ast.parse(source, mode="eval").body)


def _implicit_formula_space(compiled, max_candidates=32):
    """Factor a correct expression into a small, fully enumerated token grammar."""
    correct = next(option for option in compiled.options if option.correct)
    tokenized = [re.findall(r"C|P|\d+|[()+*/^!,-]", option.expression)
                 for option in compiled.options]
    tokens = tokenized[next(i for i, option in enumerate(compiled.options) if option.correct)]
    if "".join(tokens) != correct.expression:
        return None
    number_domain = sorted({t for row in tokenized for t in row if t.isdigit()}, key=int)
    function_domain = sorted({t for row in tokenized for t in row if t in {"C", "P"}})
    operator_domain = sorted({t for row in tokenized for t in row if t in "+-*/^"})
    slots = []
    for i, token in enumerate(tokens):
        domain = (number_domain if token.isdigit() else function_domain if token in {"C", "P"}
                  else operator_domain if token in "+-*/^" else [])
        alternatives = [x for x in domain if x != token]
        if alternatives:
            values = [token, *random.sample(alternatives, min(2, len(alternatives)))]
            random.shuffle(values)
            slots.append((i, values))
    random.shuffle(slots)
    selected, size = [], 1
    for slot in slots:
        if size * len(slot[1]) <= max_candidates:
            selected.append(slot)
            size *= len(slot[1])
        if len(selected) == 3:
            break
    if len(selected) < 2:
        return None
    selected.sort()
    candidates, values = [], []
    for replacements in product(*(domain for _, domain in selected)):
        candidate = list(tokens)
        for (index, _), replacement in zip(selected, replacements):
            candidate[index] = replacement
        expression = "".join(candidate)
        try:
            value = _evaluate_expression(expression)
        except (ArithmeticError, ValueError):
            return None
        candidates.append(expression)
        values.append(value)
    if len(candidates) < 4:
        return None
    matches = [expression for expression, value in zip(candidates, values)
               if value == correct.value]
    if matches != [correct.expression]:
        return None
    form = list(tokens)
    rules = []
    for n, (index, domain) in enumerate(selected, 1):
        name = f"X{n}"
        form[index] = name
        rules.append(f"{name} := {' | '.join(domain)}")
    return "".join(form), rules, candidates, values


def _expression_features(expression):
    depth = 0
    top_ops = []
    for char in expression:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in "+-*/^":
            top_ops.append(char)
    if "-" in top_ops:
        top = "subtraction"
    elif "+" in top_ops:
        top = "sum"
    elif "*" in top_ops:
        top = "product"
    elif "/" in top_ops:
        top = "division"
    elif "^" in top_ops:
        top = "power"
    elif expression.startswith("C("):
        top = "combination"
    elif expression.startswith("P("):
        top = "permutation"
    elif "!" in expression:
        top = "factorial"
    else:
        top = "scalar"
    literals = [int(x) for x in re.findall(r"\d+", expression)]
    return {
        "top_operator": top,
        "ast_size": len(re.findall(r"C|P|\d+|[+*^/!-]", expression)),
        "contains_combination": "C(" in expression,
        "contains_permutation": "P(" in expression,
        "contains_power": "^" in expression,
        "contains_subtraction": "-" in expression,
        "contains_factorial": "!" in expression,
        "max_literal": max(literals, default=0),
    }


class CombinatoricsFormula(Task):
    """Synthesize the unique expression in a small constrained grammar."""

    config_cls = CombinatoricsConfig

    def _sample_atomic(self):
        family = random.choices(
            ("select_arrange", "distribution", "rule", "union", "path_symmetry"),
            weights=(25, 20, 15, 10, 10),
            k=1,
        )[0]
        if family == "select_arrange":
            if random.random() < 0.8:
                n = random.randint(6, 14)
                return Select(n, random.randint(2, min(5, n - 2)), random.choice((False, True)),
                              random.choice((False, True)))
            return Arrange(random.randint(6, 11), adjacent_pair=True)
        if family == "distribution":
            boxes = random.randint(3, 6)
            return Distribute(random.randint(boxes + 2, 15), boxes, random.choice((False, True)))
        if family == "rule":
            return ChoiceRule(random.choice(("sum", "product", "complement")),
                              random.randint(4, 9), random.randint(3, 7))
        if family == "union":
            first, second = random.randint(8, 20), random.randint(8, 20)
            return UnionCount(first, second, random.randint(2, min(first, second) - 2))
        if random.random() < 0.5:
            return Arrange(random.randint(5, 11), circular=random.choice((False, True)))
        return LatticePath(random.randint(2, 8), random.randint(2, 8))

    def _sample_composed(self):
        family = random.choice(("role", "exact", "manager", "through_point", "exclusive"))
        if family == "role":
            return RoleThenCommittee(random.randint(8, 14), random.randint(2, 4))
        if family == "exact":
            length = random.randint(6, 10)
            return ExactSymbolStrings(length, random.randint(3, 7), random.randint(2, min(4, length - 2)))
        if family == "manager":
            population = random.randint(10, 16)
            return ManagerCommittee(population, random.randint(2, 4), random.randint(3, 5))
        if family == "through_point":
            right, up = random.randint(5, 9), random.randint(5, 9)
            return ThroughPointPath(right, up, random.randint(2, right - 2), random.randint(2, up - 2))
        return ExclusiveCommittee(random.randint(9, 15), random.randint(3, 5))

    def _sample_depth_3(self):
        if random.random() < 0.5:
            return ProjectChairCommittee(random.randint(2, 5), random.randint(9, 15),
                                         random.randint(2, 5))
        right, up = random.randint(7, 12), random.randint(7, 12)
        first_right, first_up = random.randint(1, right - 3), random.randint(1, up - 3)
        second_right = random.randint(first_right + 1, right - 1)
        second_up = random.randint(first_up + 1, up - 1)
        return TwoCheckpointPath(right, up, first_right, first_up, second_right, second_up)

    def _sample_program(self):
        if random.random() < self.config.depth_3_rate:
            return self._sample_depth_3()
        if random.random() < self.config.depth_2_rate:
            return self._sample_composed()
        return self._sample_atomic()

    def generate_entry(self):
        multiple_choice = random.random() < self.config.multiple_choice_prob
        for _ in range(int(self.config.max_tries)):
            program = self._sample_program()
            compiled = _compile(program, explicit=random.random() < self.config.explicit_rate)
            if not _valid(compiled):
                continue
            options = list(compiled.options)
            random.shuffle(options)
            correct_index = next(i for i, option in enumerate(options) if option.correct)
            answer = "ABCD"[correct_index] if multiple_choice else options[correct_index].expression
            space = None if multiple_choice else _implicit_formula_space(compiled)
            if not multiple_choice and space is None:
                continue
            option_rows = []
            for option in options:
                row = asdict(option)
                row["expression_features"] = _expression_features(option.expression)
                option_rows.append(row)
            correct_features = option_rows[correct_index]["expression_features"]
            metadata = edict(
                family=compiled.family,
                structural_depth=compiled.depth,
                program_type=type(program).__name__,
                program=asdict(program),
                correct_expression=options[correct_index].expression,
                correct_option_index=correct_index,
                correct_option_label="ABCD"[correct_index],
                correct_features=correct_features,
                mutation_types=[option.mutation for option in options if not option.correct],
                options=option_rows,
                multiple_choice=multiple_choice,
            )
            if space:
                answer_form, slot_rules, candidates, candidate_values = space
                metadata.update(
                    answer_form=answer_form,
                    slot_rules=slot_rules,
                    n_slots=len(slot_rules),
                    n_candidates=len(candidates),
                    candidate_expressions=candidates,
                    finite_space_exhaustively_checked=True,
                    candidate_values=[str(value) for value in candidate_values],
                )
            metadata.payload = {"problem": compiled.text}
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("failed to generate a counting expression")

    def render_prompt(self, metadata):
        if metadata.multiple_choice:
            options = "\n".join(
                f"{'ABCD'[i]}. {option['expression']}" for i, option in enumerate(metadata.options)
            )
            return (
                "Which expression counts the outcomes? C(n,k) is unordered; P(n,k) is ordered.\n\n"
                f"{render_payload(metadata.payload)}\n\nOptions:\n{options}\n"
                "Answer A-D."
            )
        rules = "\n".join(metadata.slot_rules)
        return (
            "Write the counting expression. C(n,k) is unordered; P(n,k) is ordered.\n\n"
            f"{render_payload(metadata.payload)}\n\n"
            f"The answer must have the form:\n{metadata.answer_form}\n"
            f"where:\n{rules}\n"
            "Answer with the complete expression."
        )

    def balancing_key(self, problem):
        return "|".join((
            problem.metadata.family,
            problem.metadata.correct_features.top_operator,
            problem.answer if problem.metadata.multiple_choice else str(problem.metadata.n_slots),
        ))
