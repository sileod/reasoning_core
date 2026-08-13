"""Query-aware finite-domain CSPs over a shared semantic representation."""

from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass, replace
from typing import Optional

from reasoning_core.tasks._csp_utils import (
    ALIASES, FAMILIES, CSPSolver, Eq, EqVar, Ne, consistency_metrics,
    enumeration_metrics, generate_instance, graph_relation_candidate,
    possibility_metrics, relation_metrics, split_key, value_metrics,
)
from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround


@dataclass
class ConstraintSatisfactionConfig(Config):
    n_vars: int = 2
    max_domain: int = 2
    n_constraints: int = 3
    coef_bound: int = 3
    unsat_prob: float = 0.0  # legacy override: force an UNSAT consistency example
    max_tries: int = 256

    # Compatibility aliases: attribute -> assignment, linear -> numeric.
    model_mode: str = "any"
    solve_mode: str = "query"  # query | all | min; lex_all aliases all
    max_solutions: Optional[int] = 256

    # Semantic selection controls.
    minimization_orders: int = 2
    counterfactual_prob: float = 0.15
    possibility_prob: float = 0.15
    relation_prob: float = 0.10
    consistency_prob: float = 0.20
    unsat_given_consistency: float = 0.50

    # Numeric scope topology; edge_prob also controls graph-family density.
    structure_mode: str = "any"
    max_arity: int = 3
    edge_prob: float = 0.25
    n_clusters: int = 3
    p_in: float = 0.6
    p_out: float = 0.1
    grid_width: Optional[int] = None

    def apply_difficulty(self, level):
        self.n_vars = sround(self.n_vars + 0.6 * level)
        self.max_domain = sround(self.max_domain + 0.4 * level)
        self.n_constraints = sround(self.n_constraints + 1.1 * level)
        self.coef_bound = sround(self.coef_bound + 0.3 * level)


CSPConfig = ConstraintSatisfactionConfig


class ConstraintSatisfaction(Task):
    summary = "Solve query-aware assignment, graph, scheduling, grid, set, and numeric CSPs."
    config_cls = ConstraintSatisfactionConfig

    def __init__(self, config=None):
        super().__init__(config=config or self.config_cls(), timeout=4)
        c = self.config
        if c.solve_mode.lower() == "lex_all": c.solve_mode = "all"
        modes = set(FAMILIES) | set(ALIASES) | {"any"}
        if c.model_mode.lower() not in modes: raise ValueError(f"Unknown model_mode: {c.model_mode!r}")
        if c.solve_mode.lower() not in {"query", "all", "min"}: raise ValueError(f"Unknown solve_mode: {c.solve_mode!r}")
        if c.structure_mode.lower() not in {"complete", "random", "graph", "grid", "clustered", "any"}:
            raise ValueError(f"Unknown structure_mode: {c.structure_mode!r}")
        if min(c.n_constraints, c.coef_bound, c.max_tries, c.max_arity, c.minimization_orders) <= 0:
            raise ValueError("n_constraints, coef_bound, max_tries, max_arity, and minimization_orders must be positive")
        if c.max_solutions is not None and c.max_solutions <= 0: raise ValueError("max_solutions must be positive or None")
        if c.grid_width is not None and c.grid_width <= 0: raise ValueError("grid_width must be positive or None")
        for field in ("unsat_prob", "counterfactual_prob", "possibility_prob", "relation_prob",
                      "consistency_prob", "unsat_given_consistency", "edge_prob", "p_in", "p_out"):
            if not 0 <= getattr(c, field) <= 1: raise ValueError(f"{field} must be between 0 and 1")

    def generate_entry(self):
        c, rng = self.config, random
        requested = c.model_mode.lower()
        family = rng.choices(list(FAMILIES), weights=[3, 2, 3, 2, 2, 3])[0] if requested == "any" else ALIASES.get(requested, requested)
        legacy_unsat = family == "numeric" and rng.random() < c.unsat_prob
        consistency_task = legacy_unsat or rng.random() < c.consistency_prob
        desired_unsat = legacy_unsat or (consistency_task and rng.random() < c.unsat_given_consistency)
        desired_counterfactual = not consistency_task and rng.random() < c.counterfactual_prob
        instance = generate_instance(
            family, rng, max(3, int(c.n_vars) + (1 if family != "numeric" else 0)),
            max_tries=int(c.max_tries), n_orders=int(c.minimization_orders),
            max_domain=max(2, int(c.max_domain)), coef_bound=int(c.coef_bound),
            difficulty=int(c.level), require_consistency=consistency_task,
            require_counterfactual=desired_counterfactual,
            n_constraints=int(c.n_constraints), max_arity=int(c.max_arity),
            structure_mode=c.structure_mode.lower(), edge_prob=c.edge_prob,
            n_clusters=int(c.n_clusters), p_in=c.p_in, p_out=c.p_out,
            grid_width=None if c.grid_width is None else int(c.grid_width),
        )
        clues, answer, solve_mode = list(instance.clues), instance.answer, c.solve_mode.lower()
        query, query_type = instance.query, instance.query.kind

        if desired_counterfactual:
            index, mutation, changed_value, changed_answer = instance.counterfactual_pair
            clues[index], answer, query_type = mutation, changed_answer, "counterfactual"
            query = replace(query, answer=changed_value)

        relation_info = None
        if (family == "graph" and not consistency_task and query_type != "counterfactual"
                and rng.random() < c.relation_prob):
            relation_info = graph_relation_candidate(instance.world, instance.base, clues, rng)
            if relation_info:
                answer = "Yes" if relation_info[2] else "No"
                query_type = "relation"

        possibility_value = None
        if (not consistency_task and query_type != "counterfactual"
                and query_type != "relation"
                and rng.random() < c.possibility_prob):
            active_solver = CSPSolver(instance.world.variables, instance.base)
            ambiguous = [(var, active_solver.possible_values(var, clues))
                         for var in instance.world.variables]
            ambiguous = [(var, values) for var, values in ambiguous if len(values) > 1]
            if ambiguous:
                var, values = rng.choice(ambiguous)
                impossible = [
                    value for value in var.domain if value not in values
                    and len(active_solver.formula_refutation_core(Eq(var, value), clues)) >= 2
                    and all(active_solver.is_sat((clue,), (Eq(var, value),)) for clue in clues)
                ]
                ask_possible = not impossible or rng.random() < 0.5
                possibility_value = rng.choice(values if ask_possible else impossible)
                answer = "Yes" if ask_possible else "No"
                query = replace(query, var=var)
                query_type = "possibility"

        if consistency_task:
            clues = list(instance.unsat_clues if desired_unsat else instance.sat_consistency_clues)
            answer, query_type = ("UNSAT" if desired_unsat else "SAT"), "consistency"

        # Full-assignment modes remain available for backwards compatibility, but
        # query mode is the SFT-oriented default.
        enumerated_solutions = None
        if family == "numeric" and solve_mode in {"all", "min"}:
            active_solver = CSPSolver(instance.world.variables, instance.base, clues)
            solutions, overflow = active_solver.solutions(limit=c.max_solutions) if solve_mode == "all" else (None, False)
            if overflow:
                clues += [Eq(v, instance.world.witness[v]) for v in instance.world.variables]
                solutions, overflow = CSPSolver(instance.world.variables, instance.base, clues).solutions(limit=c.max_solutions)
            if overflow: raise RuntimeError("Numeric CSP exceeded max_solutions after bounded completion")
            if solve_mode == "min":
                solution = active_solver.lex_solution()
                answer = "UNSAT" if solution is None else json.dumps(list(solution))
            elif not solutions: answer = "UNSAT"
            else: answer = json.dumps([list(x) for x in solutions])
            enumerated_solutions = solutions
            query_type = "all_solutions" if solve_mode == "all" else "lexicographic_solution"

        rendered = [formula.render(instance.renderer) for formula in clues]
        family_adapter = FAMILIES[family]
        preamble = "\n".join(filter(None, (
            family_adapter.domains_text(instance.world), family_adapter.invariant(instance.world),
        )))
        constraints_text = "\n".join(f"{i}. {text}" for i, text in enumerate(rendered, 1))
        question_text, answer_policy = query.text, "Answer with one name or integer."
        if query_type == "all_solutions":
            question_text = "Enumerate all satisfying assignments in variable order [" + ", ".join(v.name for v in instance.world.variables) + "]."
            answer_policy = "The answer is a lexicographically sorted JSON list of lists, or UNSAT."
        elif query_type == "lexicographic_solution":
            question_text = "What is the lexicographically smallest satisfying assignment in variable order [" + ", ".join(v.name for v in instance.world.variables) + "]?"
            answer_policy = "The answer is a JSON list of integers, or UNSAT."
        elif query_type == "consistency":
            question_text, answer_policy = "Are these constraints consistent?", "Answer with SAT or UNSAT."
        elif query_type == "possibility":
            if family == "assignment":
                label = instance.renderer.role(query.var)
                person = instance.world.data["people"][possibility_value]
                question_text = f"Can the {label} be {person}?"
            else:
                question_text = f"Can {query.var.name} equal {possibility_value}?"
            answer_policy = "Answer Yes or No."
        elif query_type == "relation":
            question_text = f"Must {relation_info[0].name} and {relation_info[1].name} have the same color?"
            answer_policy = "Answer Yes or No."
        prompt = f"{preamble}\n\nConstraints:\n{constraints_text}\n\nQuestion: {question_text}\n{answer_policy}"

        metric_solver = CSPSolver(instance.world.variables, instance.base)
        group_of = instance.world.data.get("group_of")
        if query_type == "counterfactual":
            metrics = value_metrics(
                metric_solver, clues, query, group_of, "counterfactual_unique_value",
            )
        elif query_type in {"scalar", "entity"}:
            metrics = dict(instance.metrics)
            essential = metrics["essential_for_query"]
            metrics.update({
                "schema": "value", "objective": "unique_value",
                "essential_for_objective": essential,
                "displayed_clue_essentiality": round(sum(essential) / len(clues), 4),
            })
        elif query_type == "possibility":
            metrics = possibility_metrics(metric_solver, clues, query.var, possibility_value, group_of)
        elif query_type == "relation":
            metrics = relation_metrics(
                metric_solver, clues, EqVar(relation_info[0], relation_info[1]),
                relation_info[2], group_of,
            )
        elif query_type == "consistency":
            metrics = consistency_metrics(metric_solver, clues, group_of)
        elif query_type in {"all_solutions", "lexicographic_solution"}:
            metrics = enumeration_metrics(
                metric_solver, clues,
                "all_solutions" if query_type == "all_solutions" else "lexicographic_solution",
                enumerated_solutions, group_of,
            )
        else:
            raise RuntimeError(f"Unknown final CSP objective: {query_type}")
        metadata = edict({
            "model_mode": requested if requested != "any" else family,
            "family": family, "solve_mode": solve_mode, "query_type": query_type,
            "structure_mode": instance.world.data.get("structure_mode"),
            "query": ([relation_info[0].name, relation_info[1].name] if relation_info else
                      (int(query.var.name.split('c')[0][1:]), int(query.var.name.split('c')[1]))
                      if family == "grid" else query.var.name),
            "query_var": None if relation_info else query.var.name,
            "clues": rendered, "constraints": rendered,
            "query_value": possibility_value,
            "query_pair": [relation_info[0].name, relation_info[1].name] if relation_info else None,
            "canonical_clues": [repr(x.canonical()) for x in clues],
            "base_constraints": [repr(x.canonical()) for x in instance.base],
            "counterfactual_candidates": [repr(x.canonical()) for x in instance.counterfactuals],
            "counterfactual_applied": query_type == "counterfactual",
            "metrics": metrics, "split_key": split_key(family, instance.base, clues, query_type),
            "solution": None if answer == "UNSAT" else answer,
            "prompt": prompt, "payload": {"instance": prompt.rsplit("\n\nQuestion:", 1)[0]},
            "render_seed": 0,
        })
        if family == "grid":
            metadata.relational_clues_required = any(" < " in x or " > " in x for x in rendered)
        return Entry(metadata=metadata, answer=str(answer))

    def render_prompt(self, metadata):
        return metadata.prompt

    def score_answer(self, answer, entry):
        expected = entry.answer if hasattr(entry, "answer") else entry["answer"]
        mode = (entry.metadata if hasattr(entry, "metadata") else entry["metadata"]).get("query_type")
        if mode not in {"all_solutions", "lexicographic_solution"}:
            return float(str(answer).strip().casefold() == str(expected).strip().casefold())
        def parse(value):
            if not isinstance(value, str): return None
            if value.strip().upper() == "UNSAT": return "UNSAT"
            try:
                result = ast.literal_eval(value.strip())
                if isinstance(result, list) and all(type(x) is int for x in result): return result
                if isinstance(result, list) and all(isinstance(row,list) and all(type(x) is int for x in row) for row in result): return result
            except (ValueError, SyntaxError): pass
            return None
        return float(parse(answer) == parse(expected))


if __name__ == "__main__":
    task = ConstraintSatisfaction()
    problem = task.generate_entry()
    print(task.render_prompt(problem.metadata)); print(problem.answer)
