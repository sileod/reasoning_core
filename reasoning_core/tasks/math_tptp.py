import networkx as nx
import re
import os
import sys
import tempfile
import random
import json
import gzip
import logging
import ast
import time
from easydict import EasyDict as edict
from dataclasses import dataclass
from appdirs import AppDirs
from pathlib import Path
from reasoning_core.utils.udocker_process import get_prover_session
from ._tptp_finite_interpretation import (
    model_is_nondegenerate,
    requirement_holds,
    requirements_hold,
    run_vampire_fmb_signed,
    serialize_model,
    universally_quantify,
    validate_formula,
)
from ._tptp_sat_graph import generate_derivation_graph
from reasoning_core.template import Task, DevTask, Entry, Config
from reasoning_core.template import TimeoutException
from itertools import combinations
from math import comb


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TPTPTerm:
    name: str
    args: tuple = ()


@dataclass(frozen=True)
class TPTPLiteral:
    sign: bool
    pred: str
    args: tuple = ()


_TPTP_TOKEN = re.compile(
    r"""\s*(?:(?P<quoted>'(?:\\.|''|[^'\\])*'|"(?:\\.|""|[^"\\])*")|"""
    r"(?P<op>!=|[()|~,=])|"
    r"(?P<word>\$?[A-Za-z0-9_]+))"
)


def _tokenize_clause(text):
    tokens = []
    pos = 0
    while pos < len(text):
        match = _TPTP_TOKEN.match(text, pos)
        if not match:
            raise ValueError(f"Unexpected TPTP syntax at position {pos}: {text[pos:pos + 20]!r}")
        tokens.append(match.group("quoted") or match.group("op") or match.group("word"))
        pos = match.end()
    return tokens


def _matching_outer_parens(tokens):
    if len(tokens) < 2 or tokens[0] != "(" or tokens[-1] != ")":
        return False
    depth = 0
    for index, token in enumerate(tokens):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses")
            if depth == 0 and index != len(tokens) - 1:
                return False
    if depth:
        raise ValueError("Unbalanced parentheses")
    return True


def _strip_outer_parens(tokens):
    while _matching_outer_parens(tokens):
        tokens = tokens[1:-1]
    return tokens


def _split_top_level(tokens, separator):
    parts = []
    start = 0
    depth = 0
    for index, token in enumerate(tokens):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses")
        elif token == separator and depth == 0:
            parts.append(tokens[start:index])
            start = index + 1
    if depth:
        raise ValueError("Unbalanced parentheses")
    parts.append(tokens[start:])
    if any(not part for part in parts):
        raise ValueError(f"Missing expression around {separator!r}")
    return parts


def _parse_term_tokens(tokens, start=0):
    if start >= len(tokens) or tokens[start] in {"(", ")", "|", "~", ",", "=", "!="}:
        raise ValueError("Expected a term")
    name = tokens[start]
    index = start + 1
    args = []
    if index < len(tokens) and tokens[index] == "(":
        index += 1
        if index < len(tokens) and tokens[index] == ")":
            return TPTPTerm(name, ()), index + 1
        while True:
            arg, index = _parse_term_tokens(tokens, index)
            args.append(arg)
            if index >= len(tokens):
                raise ValueError("Unclosed term argument list")
            if tokens[index] == ")":
                index += 1
                break
            if tokens[index] != ",":
                raise ValueError("Expected ',' or ')' in term")
            index += 1
    return TPTPTerm(name, tuple(args)), index


def _parse_complete_term(tokens):
    term, index = _parse_term_tokens(tokens)
    if index != len(tokens):
        raise ValueError("Unexpected tokens after term")
    return term


def _find_top_level_equality(tokens):
    depth = 0
    found = []
    for index, token in enumerate(tokens):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif token in {"=", "!="} and depth == 0:
            found.append((index, token))
    if len(found) > 1:
        raise ValueError("A literal may contain only one top-level equality")
    return found[0] if found else None


def _parse_literal_tokens(tokens):
    tokens = _strip_outer_parens(tokens)
    sign = True
    if tokens and tokens[0] == "~":
        sign = False
        tokens = _strip_outer_parens(tokens[1:])
    if not tokens:
        raise ValueError("Empty literal")

    equality = _find_top_level_equality(tokens)
    if equality:
        index, operator = equality
        left = _parse_complete_term(tokens[:index])
        right = _parse_complete_term(tokens[index + 1:])
        return TPTPLiteral(sign=(sign if operator == "=" else not sign), pred="=", args=(left, right))

    atom = _parse_complete_term(tokens)
    if _is_variable(atom):
        raise ValueError("A variable cannot be used as a predicate")
    return TPTPLiteral(sign=sign, pred=atom.name, args=atom.args)


def parse_clause(text):
    """Parse the first-order CNF subset emitted by E prover."""
    tokens = _strip_outer_parens(_tokenize_clause(str(text).strip()))
    if tokens == ["$false"]:
        return []
    return [_parse_literal_tokens(part) for part in _split_top_level(tokens, "|")]


def _is_variable(term):
    return not term.args and bool(re.fullmatch(r"[A-Z][A-Za-z0-9_]*", term.name))


def _walk_term_variables(term):
    if _is_variable(term):
        yield term.name
    for arg in term.args:
        yield from _walk_term_variables(arg)


def _walk_literal_variables(literal):
    for arg in literal.args:
        yield from _walk_term_variables(arg)


def _replace_variables_term(term, replacements):
    if _is_variable(term) and term.name in replacements:
        return TPTPTerm(replacements[term.name])
    return TPTPTerm(
        term.name,
        tuple(_replace_variables_term(arg, replacements) for arg in term.args),
    )


def _replace_variables_literal(literal, replacements):
    return TPTPLiteral(
        literal.sign,
        literal.pred,
        tuple(_replace_variables_term(arg, replacements) for arg in literal.args),
    )


def rename_apart(clause, prefix="Y", forbidden=()):
    forbidden = set(forbidden)
    replacements = {}
    next_index = 1
    for literal in clause:
        for variable in _walk_literal_variables(literal):
            if variable in replacements:
                continue
            candidate = f"{prefix}{next_index}"
            while candidate in forbidden:
                next_index += 1
                candidate = f"{prefix}{next_index}"
            replacements[variable] = candidate
            forbidden.add(candidate)
            next_index += 1
    return [_replace_variables_literal(literal, replacements) for literal in clause]


def _dereference(term, subst):
    seen = set()
    while _is_variable(term) and term.name in subst and term.name not in seen:
        seen.add(term.name)
        term = subst[term.name]
    return term


def _occurs(variable, term, subst):
    term = _dereference(term, subst)
    if _is_variable(term):
        return term.name == variable
    return any(_occurs(variable, arg, subst) for arg in term.args)


def unify(left, right, subst=None):
    """Return an MGU for two terms, with an occurs check, or None."""
    subst = {} if subst is None else dict(subst)
    pending = [(left, right)]
    while pending:
        first, second = pending.pop()
        first = _dereference(first, subst)
        second = _dereference(second, subst)
        if first == second:
            continue
        if _is_variable(first):
            if _occurs(first.name, second, subst):
                return None
            subst[first.name] = second
            continue
        if _is_variable(second):
            if _occurs(second.name, first, subst):
                return None
            subst[second.name] = first
            continue
        if first.name != second.name or len(first.args) != len(second.args):
            return None
        pending.extend(zip(first.args, second.args))
    return subst


def _unify_args(left_args, right_args):
    if len(left_args) != len(right_args):
        return None
    subst = {}
    for left, right in zip(left_args, right_args):
        subst = unify(left, right, subst)
        if subst is None:
            return None
    return subst


def apply_subst_term(term, subst):
    term = _dereference(term, subst)
    if _is_variable(term):
        return term
    return TPTPTerm(
        term.name,
        tuple(apply_subst_term(arg, subst) for arg in term.args),
    )


def apply_subst_literal(literal, subst):
    return TPTPLiteral(
        literal.sign,
        literal.pred,
        tuple(apply_subst_term(arg, subst) for arg in literal.args),
    )


def resolvents(clause_a, clause_b):
    out = []
    for index_a, literal_a in enumerate(clause_a):
        for index_b, literal_b in enumerate(clause_b):
            if literal_a.sign == literal_b.sign or literal_a.pred != literal_b.pred:
                continue
            subst = _unify_args(literal_a.args, literal_b.args)
            if subst is None:
                continue
            remaining = [
                apply_subst_literal(literal, subst)
                for index, literal in enumerate(clause_a)
                if index != index_a
            ] + [
                apply_subst_literal(literal, subst)
                for index, literal in enumerate(clause_b)
                if index != index_b
            ]
            out.append(list(dict.fromkeys(remaining)))
    return out


def _render_term(term, variable_mask=None):
    name = variable_mask if variable_mask is not None and _is_variable(term) else term.name
    if not term.args:
        return name
    return f"{name}({','.join(_render_term(arg, variable_mask) for arg in term.args)})"


def _render_literal(literal, variable_mask=None):
    if literal.pred == "=":
        operator = "=" if literal.sign else "!="
        return (
            f"{_render_term(literal.args[0], variable_mask)} {operator} "
            f"{_render_term(literal.args[1], variable_mask)}"
        )
    atom = literal.pred
    if literal.args:
        atom += f"({','.join(_render_term(arg, variable_mask) for arg in literal.args)})"
    return atom if literal.sign else f"~{atom}"


def render_clause(clause):
    if not clause:
        return "$false"
    return f"({' | '.join(_render_literal(literal) for literal in clause)})"


def canonical(clause):
    """Sort literals and alpha-normalize variables, rejecting order ties."""
    if not clause:
        return "$false"
    masked = [_render_literal(literal, variable_mask="X") for literal in clause]
    if len(set(masked)) != len(masked):
        return None
    ordered = [literal for _, literal in sorted(zip(masked, clause), key=lambda item: item[0])]
    replacements = {}
    for literal in ordered:
        for variable in _walk_literal_variables(literal):
            if variable not in replacements:
                replacements[variable] = f"X{len(replacements) + 1}"
    return render_clause([_replace_variables_literal(literal, replacements) for literal in ordered])


def _term_depth(term):
    if not term.args:
        return 0
    return 1 + max(_term_depth(arg) for arg in term.args)


def clause_term_depth(clause):
    return max(
        (_term_depth(arg) for literal in clause for arg in literal.args),
        default=0,
    )


def _strip_answer_ticks(answer):
    answer = str(answer).strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer[3:-3].strip()
        if answer.startswith("tptp"):
            answer = answer[4:].lstrip()
    return answer.strip().strip("`").strip()


def _inference_rule(inference):
    inference = inference or ""
    match = re.search(r"inference\(([A-Za-z0-9_]+)", inference)
    if match:
        return match.group(1)
    match = re.match(r"([A-Za-z0-9_]+)", inference)
    return match.group(1) if match else "inference"


def extract_problem_from_graph(G: nx.DiGraph, node_id_str: str, max_length_proof: int):
    theorem = G.nodes[node_id_str]['data'].clause_formula
    frontier = {node_id_str}
    collected_hypotheses = set()
    
    for _ in range(max_length_proof):
        nxt = set()
        for v in frontier:
            parents = list(G.predecessors(v))
            if parents:
                # Continue traversing up the graph
                nxt.update(parents)
            else:
                # FIX: Capture leaves (axioms) encountered on short branches
                collected_hypotheses.add(v)

        if not nxt:
            break
        frontier = nxt
    
    # The final frontier (nodes at max_depth) are also hypotheses
    collected_hypotheses.update(frontier)
    
    hypotheses = [G.nodes[n]['data'].clause_formula for n in collected_hypotheses]
    hypotheses = [h for h in hypotheses if normalize_formula(h) != normalize_formula(theorem)]
    return hypotheses, theorem

def extract_useful_axioms(G: nx.DiGraph, node_id_str: str) : 
    ancestors = nx.ancestors(G, node_id_str)

    initial_ax = {n for n, in_degree in G.in_degree() if in_degree == 0}

    useful_ax = ancestors.intersection(initial_ax)

    return useful_ax


def _node_is_background(node):
    role = (node.role or "").lower()
    clause_id = (node.clause_id or "").lower()
    if "conjecture" in role or "hypothesis" in role:
        return False
    if re.search(r"(goal|conj|theorem|lemma|prove|query|negated)", clause_id):
        return False
    if role in {"axiom", "definition", "type"}:
        return True
    return bool(node.inference and node.inference.startswith("file("))


def _node_sort_key(node_id):
    return (0, int(node_id)) if str(node_id).isdigit() else (1, str(node_id))


def split_problem_from_proof(G: nx.DiGraph, target, is_background=_node_is_background, return_nodes=False):
    useful = extract_useful_axioms(G, target)
    background, premises = [], []
    background_nodes, premise_nodes = [], []
    used_split_fallback = False
    theorem = G.nodes[target]["data"].clause_formula
    theorem_norm = normalize_formula(theorem)
    leaf_items = []
    for n in sorted(useful, key=_node_sort_key):
        node = G.nodes[n]["data"]
        formula = node.clause_formula
        if normalize_formula(formula) == theorem_norm:
            continue
        leaf_items.append((n, formula, bool(is_background(node))))

    if leaf_items and (all(item[2] for item in leaf_items) or not any(item[2] for item in leaf_items)):
        if len(leaf_items) < 3:
            leaf_items = []
        else:
            used_split_fallback = True
            random.shuffle(leaf_items)
            background_count = random.randint(1, len(leaf_items) - 1)
            background_ids = {n for n, _, _ in leaf_items[:background_count]}
            leaf_items = [(n, formula, n in background_ids) for n, formula, _ in leaf_items]

    if leaf_items and (not any(item[2] for item in leaf_items) or all(item[2] for item in leaf_items)):
        leaf_items = []

    for n, formula, is_bg in sorted(leaf_items, key=lambda item: _node_sort_key(item[0])):
        if is_bg:
            background_nodes.append(n)
            background.append(formula)
        else:
            premise_nodes.append(n)
            premises.append(formula)
    background = list(dict.fromkeys(background))
    premises = list(dict.fromkeys(premises))
    if return_nodes:
        return background, premises, theorem, background_nodes, premise_nodes, used_split_fallback
    return background, premises, theorem


def normalize_formula(f: str) -> str:
    """Canonicalize formula: remove whitespace and anonymize variables."""
    if not f: return ""
    # Remove whitespace
    f = re.sub(r"\s+", "", f)
    # Replace variables (e.g., X123) with generic V to handle alpha-equivalence
    f = re.sub(r"X\d+", "V", f)
    return f

# 2. FIX: Clean CoT generation with step collapsing and better labeling
def make_cot(G: nx.DiGraph, target_node: str, formula_map: dict) -> str:
    sub = G.subgraph(nx.ancestors(G, target_node) | {target_node})
    lines = []
    node_to_label = {}
    step_counter = 0
    sys_ax_counter = 0  # Fixed variable name

    # Topological sort ensures we process parents before children
    for node in nx.topological_sort(sub):
        
        # Optimization: Skip intermediate 1-parent nodes (normalization/copy steps)
        # This removes "c_0_X" noise unless it's the final theorem
        parents = sorted(list(sub.predecessors(node)))
        if len(parents) == 1 and node != target_node:
            # Inherit label from the single parent (collapse step)
            p_lbl = node_to_label.get(parents[0])
            if p_lbl:
                node_to_label[node] = p_lbl
                continue

        data = sub.nodes[node]['data']
        f_norm = normalize_formula(data.clause_formula)
        
        val = formula_map.get(f_norm)
        is_theorem = (node == target_node)
        
        # Determine Label
        if not parents:
            # Leaf / Axiom
            if is_theorem:
                label = "THEOREM"
                lines.append(f"THEOREM [ '{data.clause_formula.strip()}' ] (axiom)")
            elif val is not None and str(val) != "THEOREM":
                label = f"premise_{val}"
            else:
                # Fallback for unmapped system axioms
                label = f"sys_ax_{sys_ax_counter}"
                sys_ax_counter += 1
            node_to_label[node] = label
            continue

        # Derived Node
        # Get parent labels
        p_labels = [node_to_label.get(p) for p in parents if p in node_to_label]
        if not p_labels: continue

        if is_theorem:
            label = "THEOREM"
        else:
            label = f"step_{step_counter}"
            step_counter += 1
        
        node_to_label[node] = label

        # Clean Inference Rule Name
        # Extract 'res', 'pm', 'rw' from string like "inference(rw,[status...])"
        inf_str = data.inference or ""
        rule_match = re.search(r'inference\(([a-zA-Z0-9_]+)', inf_str)
        if rule_match:
            rule_name = rule_match.group(1)
        else:
            # Fallback cleanup for non-standard formats
            rule_name = re.match(r'([a-zA-Z0-9_]+)', inf_str).group(1) if inf_str else "inference"
            if rule_name.startswith("c_0"): rule_name = "processing"

        lines.append(f"{label} {rule_name}({', '.join(p_labels)}): [ '{data.clause_formula.strip()}' ]")

    return "\n".join(lines).strip()

def perturb_list(input_l: list, base_domain: list, n_perturbations: int = 1) -> list:
    """Applies cumulative perturbations to a list."""
    lst = list(input_l) 
    base_set = set(base_domain)

    for _ in range(n_perturbations):
        complementary = base_set - set(lst)
        
        possible_ops = []
        if complementary:
            possible_ops.append('add')
            if lst: 
                possible_ops.append('replace')
        if len(lst) > 1:
            possible_ops.append('remove')
        if not possible_ops:
            break
            
        op_type = random.choice(possible_ops)
        
        if op_type == 'add':
            lst.insert(random.randint(0, len(lst)), random.choice(list(complementary)))
        elif op_type == 'remove':
            lst.pop(random.randint(0, len(lst) - 1))
        elif op_type == 'replace':
            index_to_replace = random.randint(0, len(lst) - 1)
            lst[index_to_replace] = random.choice(list(complementary))
            
    return lst


def symbols_of_formula(formula):
    return {
        token for token in re.findall(r"[a-z][A-Za-z0-9_]*", str(formula))
        if not token.startswith("X")
    }


def same_signature_pool(pool, reference, min_overlap=0.7):
    ref = set().union(*(symbols_of_formula(f) for f in reference))
    if not ref:
        return []
    out = []
    for formula in pool:
        symbols = symbols_of_formula(formula)
        if symbols and len(symbols & ref) / len(symbols | ref) >= min_overlap:
            out.append(formula)
    return out


def valid_clause_list(clauses):
    try:
        for clause in clauses:
            parse_clause(clause)
    except Exception:
        return False
    return True


def alpha_key(f):
    """Alpha-normalize TPTP variables without merging distinct variables."""
    seen = {}
    f = re.sub(r"\s+", "", str(f))

    def repl(m):
        x = m.group()
        if x not in seen:
            seen[x] = f"V{len(seen)}"
        return seen[x]

    return re.sub(r"\b[A-Z][A-Za-z0-9_]*\b", repl, f)


def _clause_set_key(clauses, background=()):
    return (
        tuple(sorted(alpha_key(c) for c in background)),
        tuple(sorted(alpha_key(c) for c in clauses)),
    )


def prove_conjecture(axioms: list[str], conjecture: str,
                        time_limit_seconds: str ="30", verb: bool = False,
                        disprove_first: bool = False,
                        disprove_time_limit_seconds = None,
                        log_errors: bool = True,
                        background=()):
    """
    Uses Vampire to prove or disprove a conjecture given a set of axioms.
    Returns True (provable), False (disprovable/countersatisfiable), or an error string.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.p') as temp_f:
        for i, axiom in enumerate(background, 1):
            temp_f.write(f"cnf(background_{i}, axiom, {axiom}).\n")
        for i, axiom in enumerate(axioms, 1):
            temp_f.write(f"cnf(axiom_{i}, axiom, {axiom}).\n")
        temp_f.write(f"fof(conjecture_1, conjecture, {universally_quantify(conjecture)}).\n")
        temp_f.flush()
        
        if verb == True:
            print(f"---- proof file :-------------------------")
            temp_f.seek(0)  
            print(temp_f.read()) 
            print("-------------------------------------------------")


        prove_limit = str(time_limit_seconds)
        disprove_limit = str(disprove_time_limit_seconds or time_limit_seconds)
        vampire_command_proove = ["-t", prove_limit]
        vampire_command_disproove = ["-t", disprove_limit, "-sa", "fmb"]

        result_proove = None
        result_disproove = None

        if disprove_first:
            result_disproove = get_prover_session().run_prover('vampire', vampire_command_disproove, temp_f.name)

            if verb == True:
                print(f"output disproove vampire :  {result_disproove.stdout} ")

            if "Finite Model Found!" in result_disproove.stdout or "% SZS status CounterSatisfiable" in result_disproove.stdout:
                return False

        result_proove = get_prover_session().run_prover('vampire',vampire_command_proove,temp_f.name)

        if verb == True:
            print(f"output proove vampire :  {result_proove.stdout} ")

        if "% SZS status Theorem" in result_proove.stdout :
            return True
        if "% SZS status CounterSatisfiable" in result_proove.stdout :
            return False

        if result_disproove is None:
            result_disproove = get_prover_session().run_prover('vampire',vampire_command_disproove,temp_f.name)
    
        if verb == True:
            print(f"output disproove vampire :  {result_disproove.stdout} ")

        if "Finite Model Found!" in result_disproove.stdout or "% SZS status CounterSatisfiable" in result_disproove.stdout:
            return False
        if "% Time limit reached!" in result_proove.stdout and "% Time limit reached!" in result_disproove.stdout  :
            return f"ERROR : TIME LIMIT in both tentative to proove AND to disproove"
        if log_errors:
            print(f"[prove_conjecture] vampire failed:"
                  f"\n  prove: rc={result_proove.returncode} stdout={result_proove.stdout[:200]!r} stderr={result_proove.stderr[:200]!r}"
                  f"\n  disprove: rc={result_disproove.returncode} stdout={result_disproove.stdout[:200]!r} stderr={result_disproove.stderr[:200]!r}",
                  file=sys.stderr)
        return f"ERROR : {result_proove.stderr}{result_disproove.stderr}"


def check_clause_set_satisfiability(clauses, time_limit_seconds="5", log_errors=True, background=()):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".p") as temp_f:
        for i, clause in enumerate(background, 1):
            temp_f.write(f"cnf(bg{i}, axiom, {clause}).\n")
        for i, clause in enumerate(clauses, 1):
            temp_f.write(f"cnf(c{i}, axiom, {clause}).\n")
        temp_f.flush()

        sat = get_prover_session().run_prover(
            "vampire", ["-t", str(time_limit_seconds), "-sa", "fmb"], temp_f.name
        )
        if (
            "Finite Model Found!" in sat.stdout
            or "% SZS status Satisfiable" in sat.stdout
            or "% SZS status CounterSatisfiable" in sat.stdout
        ):
            return True

        unsat = get_prover_session().run_prover(
            "vampire", ["-t", str(time_limit_seconds)], temp_f.name
        )
        if "% SZS status Unsatisfiable" in unsat.stdout or "% SZS status Theorem" in unsat.stdout:
            return False

        if log_errors:
            print(
                "[check_clause_set_satisfiability] inconclusive:"
                f"\n  sat stdout={sat.stdout[:200]!r} stderr={sat.stderr[:200]!r}"
                f"\n  unsat stdout={unsat.stdout[:200]!r} stderr={unsat.stderr[:200]!r}",
                file=sys.stderr,
            )
        return None


def tptp_surface_features(entry):
    metadata = entry.metadata if hasattr(entry, "metadata") else entry
    clauses = list(metadata.get("background", [])) + list(metadata.get("hypotheses", []))
    clauses += list(metadata.get("clauses", []))
    if metadata.get("conjecture"):
        clauses.append(metadata["conjecture"])
    if metadata.get("negated_theorem"):
        clauses.append(metadata["negated_theorem"])
    text = "\n".join(map(str, clauses))
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|!=|[=|~(),]", text)
    counts = {
        "chars": len(text),
        "tokens": len(tokens),
        "clauses": len(clauses),
        "literals": text.count("|") + len(clauses),
        "equals": text.count("="),
        "negations": text.count("~"),
    }
    for token in tokens:
        if token and token[0].islower():
            counts[f"sym:{token}"] = counts.get(f"sym:{token}", 0) + 1
    return counts


def score_tptp_surface_baseline(entries, folds=5, model="logistic"):
    """Cross-validate a bag-of-symbols baseline on generated TPTP examples."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the TPTP surface baseline") from exc

    entries = list(entries)
    labels = [str(entry.answer).strip().lower() for entry in entries]
    if len(set(labels)) < 2:
        raise ValueError("surface baseline needs at least two answer classes")
    if model == "forest":
        estimator = RandomForestClassifier(n_estimators=100, random_state=0)
    else:
        estimator = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(DictVectorizer(), estimator)
    scores = cross_val_score(pipeline, [tptp_surface_features(e) for e in entries], labels, cv=folds)
    return {"accuracy": float(scores.mean()), "std": float(scores.std()), "n": len(entries)}


dirs = AppDirs("Axioms_TPTP")
BASE_DIR = Path(__file__).resolve().parent.parent
AXIOM_ARCHIVE_PATH = BASE_DIR / "resources" / "axioms_filtered.json.gz"
DOMAIN_MAP = {
    'ALG': 'Algebra',
    'ANA': 'Analysis',
    'FLD': 'Field Theory',
    'GEO': 'Geometry',
    'GRP': 'Group Theory',
    'LCL': 'Logic Calculi',
    'NUM': 'Number Theory',
    'RNG': 'Ring Theory',
    'SET': 'Set Theory',
    'TOP': 'Topology'
}

def get_random_tptp_axioms(
    axiom_archive=AXIOM_ARCHIVE_PATH,
    prefixes=None,
    cache_dir=dirs.user_cache_dir ):

    try:
        with gzip.open(axiom_archive, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, EOFError):
        return None, None

    keys = list(data.keys())
    if prefixes:
        keys = [k for k in keys if k.startswith(tuple(prefixes))]

    if not keys:
        return None, None
        
    chosen_key = random.choice(keys)
    content = data[chosen_key]

    try:
        os.makedirs(cache_dir, exist_ok=True)
        tempfile.TemporaryFile(dir=cache_dir).close()
    except OSError:
        cache_dir = tempfile.gettempdir()

    temp_file = tempfile.NamedTemporaryFile(
        mode='w+', 
        encoding='utf-8', 
        suffix='.p', 
        dir=cache_dir,
        delete=False  
    )
    
    with temp_file:
        temp_file.write(content)
        temp_file.flush()

    return temp_file.name, chosen_key

@dataclass
class EntailConfig(Config):
    proof_depth: int = 1
    perturbation: int = 1
    max_hypotheses: int = 8
    max_payload_chars: int = 2400
    min_interesting_score: float = 0.6
    positive_problem_ratio: float = 0.25
    max_graph_attempts: int = 20
    max_attempts: int = 80
    domains = ['ALG', 'ANA', 'FLD', 'GEO', 'GRP', 'LCL', 'NUM', 'RNG', 'SET', 'TOP']

    def apply_difficulty(self, level):
        self.proof_depth += level
        self.perturbation += level
        self.max_hypotheses += level
        self.max_payload_chars += 500 * level

class TptpEntailment(DevTask):
    """
    A task that generates problems to determine if a set of hypotheses
    proves a given conjecture.
    """
    summary = "Decide whether generated TPTP hypotheses entail conjectures across algebraic, numeric, geometric, logical, set, and topological domains."

    def __init__(self, config=None):
        super().__init__(config=config or EntailConfig())
        # Initialize prover session at task init (pulls docker image if needed)
        # This ensures docker setup happens before any generation timing
        from reasoning_core.utils.udocker_process import initialize_prover_session
        initialize_prover_session()
        self._sat_cache = {}
        self._prove_cache = {}
        self.graph = nx.DiGraph()
        self.axiom_set = None
        self.all_formulas = []
        self.interesting_thm = []

    def _sat(self, clauses, time_limit="8", background=()):
        key = (str(time_limit), *_clause_set_key(clauses, background))
        if key not in self._sat_cache:
            self._sat_cache[key] = check_clause_set_satisfiability(
                list(clauses),
                time_limit_seconds=str(time_limit),
                log_errors=False,
                background=list(background),
            )
        return self._sat_cache[key]

    def _prove(self, axioms, theorem, time_limit="15", background=(), disprove_first=False):
        key = (
            str(time_limit),
            bool(disprove_first),
            tuple(sorted(alpha_key(c) for c in background)),
            tuple(sorted(alpha_key(c) for c in axioms)),
            alpha_key(theorem),
        )
        if key not in self._prove_cache:
            self._prove_cache[key] = prove_conjecture(
                list(axioms),
                theorem,
                time_limit_seconds=str(time_limit),
                background=list(background),
                disprove_first=disprove_first,
                log_errors=False,
            )
        return self._prove_cache[key]

    def _initialize_graph(self):
        for _ in range(self.config.max_graph_attempts):
            axiom_file_path, axiom_file_name = get_random_tptp_axioms(prefixes=self.config.domains)

            if not axiom_file_path:
                continue
            try:
                self.axiom_set = axiom_file_name
                self.graph = generate_derivation_graph(
                        axiom_file = axiom_file_path,
                        save_output=False,
                        ranking=True,
                        e_limit=2
                    )
            finally:
                if os.path.exists(axiom_file_path):
                    os.remove(axiom_file_path)
            

            self.all_formulas = [data['data'].clause_formula for _, data in self.graph.nodes(data=True)]
            self.interesting_thm = []

            for i in self.graph.nodes() : 
                if (
                    self.graph.nodes[i]['data'].interesting_score <= self.config.min_interesting_score
                    or self.graph.in_degree(i) <= 1
                ):
                    continue
                try:
                    background, premises, theorem = split_problem_from_proof(self.graph, i)
                except Exception:
                    continue
                if not background or not premises:
                    continue
                if len(premises) > self.config.max_hypotheses:
                    continue
                if sum(map(len, background + premises)) + len(theorem) > self.config.max_payload_chars:
                    continue
                if not valid_clause_list(background + premises + [theorem]):
                    continue
                self.interesting_thm.append(i)
            if len(self.interesting_thm) >= 5 :
                return True
        return False

    def generate_entry(self):
        main_limit = "15"
        ablation_limit = "15"

        for _ in range(3):
            if not self.interesting_thm and not self._initialize_graph():
                raise RuntimeError("failed to build a TPTP proof graph with candidate theorem nodes")
            candidates = list(self.interesting_thm)
            random.shuffle(candidates)
            for theorem_node_id in candidates[: self.config.max_attempts]:
                (
                    background,
                    correct_hypotheses,
                    theorem,
                    background_nodes,
                    premise_nodes,
                    used_split_fallback,
                ) = split_problem_from_proof(
                    self.graph, theorem_node_id, return_nodes=True
                )
                if not background or not correct_hypotheses:
                    continue
                if not valid_clause_list(background + correct_hypotheses + [theorem]):
                    continue
                if self._sat(
                    correct_hypotheses,
                    time_limit="8",
                    background=background,
                ) is not True:
                    continue
                if self._prove(
                    correct_hypotheses,
                    theorem,
                    time_limit=ablation_limit,
                    background=[],
                    disprove_first=True,
                ) is not False:
                    continue
                if random.random() < self.config.positive_problem_ratio:
                    hypotheses = correct_hypotheses
                    if (
                        len(hypotheses) > self.config.max_hypotheses
                        or sum(map(len, background + hypotheses)) + len(theorem) > self.config.max_payload_chars
                    ):
                        continue
                    try:
                        if self._prove(
                            hypotheses, theorem, time_limit=main_limit, background=background
                        ) is not True:
                            continue
                    except TimeoutError:
                        continue
                    answer = True
                else:
                    broad_pool = set(self.all_formulas) - set(background) - set(correct_hypotheses) - {theorem}
                    distraction_pool = same_signature_pool(
                        broad_pool,
                        background + correct_hypotheses + [theorem],
                    )
                    if not distraction_pool:
                        continue
                    hypotheses = perturb_list(correct_hypotheses, distraction_pool, self.config.perturbation)
                    if (
                        len(hypotheses) > self.config.max_hypotheses
                        or sum(map(len, background + hypotheses)) + len(theorem) > self.config.max_payload_chars
                    ):
                        continue
                    try:
                        answer = self._prove(
                            hypotheses,
                            theorem,
                            time_limit=main_limit,
                            background=background,
                            disprove_first=True,
                        )
                    except TimeoutError:
                        continue
                    if answer is not False:
                        continue

                if isinstance(answer, bool):
                    metadata = edict({'background': background,
                                'background_nodes': background_nodes,
                                'hypotheses': hypotheses,
                                'premise_nodes': premise_nodes,
                                'theorem_node': theorem_node_id,
                                'used_split_fallback': used_split_fallback,
                                'conjecture': theorem,
                                'correct_hypotheses': correct_hypotheses,
                                'proof_depth': self.config.proof_depth,
                                'perturbation': self.config.perturbation,
                                'axiom_set': self.axiom_set})
                    return Entry(metadata, str(answer))
            self.interesting_thm = []
        raise RuntimeError("failed to build a compact TPTP entailment task with useful background")

    def render_prompt(self, metadata):

        background_text = "\n".join([f"- {h}" for h in metadata.get('background', [])])
        hypotheses_text = "\n".join([f"- {h}" for h in metadata['hypotheses']])

        return (
            f"Decide if the premises entail the conjecture.\n\n"
            f"TPTP source: {metadata['axiom_set']}\n\n"
            f"Background axioms:\n{background_text}\n\n"
            f"Premises:\n{hypotheses_text}\n\n"
            f"Conjecture: `{metadata['conjecture']}`\n\n"
            f"The answer is `True` (provable) or `False` (not provable)."
        )
    
    def score_answer(self, answer, entry):
        ref = entry.answer.lower()
        pred = str(answer).lower().strip().strip('"').strip("'")
        return float(ref==pred)


def negate_clause_formula(formula):
    try:
        clause = parse_clause(formula)
    except (TypeError, ValueError):
        return None
    if len(clause) != 1:
        return None
    lit = clause[0]
    return render_clause([TPTPLiteral(not lit.sign, lit.pred, lit.args)])


@dataclass
class ConsistencyRepairConfig(Config):
    proof_depth: int = 1
    perturbation: int = 1
    max_axioms: int = 8
    max_payload_chars: int = 2400
    min_interesting_score: float = 0.6
    quick_sat_time_limit: str = "5"
    sat_time_limit: str = "5"
    unsat_time_limit: str = "8"
    max_graph_attempts: int = 12
    max_attempts: int = 120
    max_split_attempts: int = 5
    graph_time_budget: float = 90.0
    generation_time_budget: float = 600.0
    answer_style: str = "space"  # "space" or "list"
    domains = ['ALG', 'ANA', 'FLD', 'GEO', 'GRP', 'LCL', 'NUM', 'RNG', 'SET', 'TOP']

    def apply_difficulty(self, level):
        self.proof_depth += level
        self.perturbation += level
        self.max_axioms += level
        self.max_payload_chars += 500 * level

class TPTPConsistencyRepair(DevTask):
    """Find all singleton deletions that restore satisfiability."""
    summary = "Find every singleton axiom deletion that restores satisfiability across generated multi-domain TPTP theories."

    def __init__(self, config=None):
        super().__init__(config=config or ConsistencyRepairConfig(), timeout=720)
        self._sat_cache = {}
        self._prove_cache = {}
        self.graph = nx.DiGraph()
        self.axiom_set = None
        self.all_formulas = []
        self.leaf_formulas = []
        self.interesting_thm = []
        from reasoning_core.utils.udocker_process import initialize_prover_session
        initialize_prover_session()
        for _ in range(self.config.max_graph_attempts):
            if self._initialize_graph():
                break

    def _sat(self, clauses, time_limit=None, background=(), log_errors=False):
        time_limit = str(time_limit or self.config.sat_time_limit)
        key = (time_limit, *_clause_set_key(clauses, background))
        if key not in self._sat_cache:
            self._sat_cache[key] = check_clause_set_satisfiability(
                list(clauses),
                time_limit,
                log_errors=log_errors,
                background=list(background),
            )
        return self._sat_cache[key]

    def _prove(self, axioms, theorem, time_limit=None, background=(), disprove_first=False):
        time_limit = str(time_limit or self.config.sat_time_limit)
        key = (
            time_limit,
            bool(disprove_first),
            tuple(sorted(alpha_key(c) for c in background)),
            tuple(sorted(alpha_key(c) for c in axioms)),
            alpha_key(theorem),
        )
        if key not in self._prove_cache:
            self._prove_cache[key] = prove_conjecture(
                list(axioms),
                theorem,
                time_limit_seconds=time_limit,
                background=list(background),
                disprove_first=disprove_first,
                log_errors=False,
            )
        return self._prove_cache[key]

    def _initialize_graph(self):
        axiom_file_path, axiom_file_name = get_random_tptp_axioms(prefixes=self.config.domains)
        if not axiom_file_path:
            return False
        try:
            self.axiom_set = axiom_file_name
            self.graph = generate_derivation_graph(
                axiom_file=axiom_file_path,
                save_output=False,
                ranking=False,
                e_limit=2,
            )
        finally:
            if os.path.exists(axiom_file_path):
                os.remove(axiom_file_path)

        self.all_formulas = [data["data"].clause_formula for _, data in self.graph.nodes(data=True)]
        self.leaf_formulas = [
            self.graph.nodes[node]["data"].clause_formula
            for node, in_degree in self.graph.in_degree()
            if in_degree == 0
        ]
        rough_candidates = [node for node in self.graph.nodes() if self.graph.in_degree(node) > 1]
        random.shuffle(rough_candidates)
        self.interesting_thm = []
        for node in rough_candidates[: max(self.config.max_attempts * 3, 60)]:
            if len(self.interesting_thm) >= self.config.max_attempts:
                break
            try:
                formulas, theorem = self._useful_leaf_formulas(node)
            except Exception:
                continue
            if len(formulas) < 2 or len(formulas) > self.config.max_axioms + 4:
                continue
            if negate_clause_formula(theorem) is None:
                continue
            if sum(map(len, formulas + [theorem])) > self.config.max_payload_chars:
                continue
            if not valid_clause_list(formulas + [theorem]):
                continue
            self.interesting_thm.append(node)
        return bool(self.interesting_thm)

    def _useful_leaf_formulas(self, theorem_node_id):
        theorem = self.graph.nodes[theorem_node_id]["data"].clause_formula
        theorem_norm = normalize_formula(theorem)
        nodes = list(extract_useful_axioms(self.graph, theorem_node_id))
        formulas = []
        for node in nodes:
            formula = self.graph.nodes[node]["data"].clause_formula
            if normalize_formula(formula) != theorem_norm:
                formulas.append(formula)
        return list(dict.fromkeys(formulas)), theorem

    def _validated_split_from_proof(self, theorem_node_id):
        formulas, theorem = self._useful_leaf_formulas(theorem_node_id)
        neg_theorem = negate_clause_formula(theorem)
        if neg_theorem is None or len(formulas) < 2:
            return None
        if not valid_clause_list(formulas + [theorem, neg_theorem]):
            return None
        random.shuffle(formulas)
        split_points = list(range(1, len(formulas)))
        random.shuffle(split_points)
        for k in split_points[: self.config.max_split_attempts]:
            background = list(dict.fromkeys(formulas[:k]))
            clauses = list(dict.fromkeys(formulas[k:]))
            if not background or not clauses:
                continue
            if len(clauses) > self.config.max_axioms:
                continue
            if sum(map(len, background + clauses + [neg_theorem])) > self.config.max_payload_chars:
                continue
            if self._sat(clauses, self.config.sat_time_limit, background=background) is not True:
                continue
            if self._sat([neg_theorem], self.config.sat_time_limit, background=background) is not True:
                continue
            if self._prove(
                clauses,
                theorem,
                self.config.sat_time_limit,
                background=[],
                disprove_first=True,
            ) is not False:
                continue
            return background, clauses, theorem, neg_theorem
        return None

    def _add_signature_distractors(self, background, clauses, theorem, neg_theorem):
        clauses = list(dict.fromkeys(clauses))
        target_total = random.randint(max(3, len(clauses)), self.config.max_axioms)
        pool = same_signature_pool(
            set(self.leaf_formulas) - set(background) - set(clauses) - {theorem, neg_theorem},
            background + clauses + [theorem, neg_theorem],
            min_overlap=0.5,
        )
        random.shuffle(pool)
        for candidate in pool:
            if len(clauses) >= target_total:
                break
            trial = clauses + [candidate]
            if len(trial) > self.config.max_axioms:
                break
            if sum(map(len, background + trial + [neg_theorem])) > self.config.max_payload_chars:
                continue
            if not valid_clause_list([candidate]):
                continue
            if self._sat(
                [candidate],
                self.config.quick_sat_time_limit,
                background=background,
            ) is not True:
                continue
            clauses.append(candidate)
        return clauses

    def _repair_indices(self, background, clauses, neg_theorem, time_limit):
        repairs = []
        for i in range(len(clauses)):
            reduced = clauses[:i] + clauses[i + 1:] + [neg_theorem]
            sat = self._sat(reduced, time_limit, background=background)
            if sat is None:
                return None
            if sat is True:
                repairs.append(i + 1)
        return repairs

    def _final_recheck_repair(self, background, clauses, neg_theorem, repairs):
        if self._sat([neg_theorem], self.config.sat_time_limit, background=background) is not True:
            return False
        if self._sat(clauses, self.config.sat_time_limit, background=background) is not True:
            return False
        if any(
            self._sat([c], self.config.sat_time_limit, background=background) is not True
            for c in clauses
        ):
            return False
        if self._sat(
            clauses + [neg_theorem],
            self.config.unsat_time_limit,
            background=background,
        ) is not False:
            return False
        return self._repair_indices(
            background, clauses, neg_theorem, self.config.sat_time_limit
        ) == repairs

    def _build_problem(self):
        deadline = time.time() + self.config.generation_time_budget
        while time.time() <= deadline:
            if not self.interesting_thm and not self._initialize_graph():
                break
            graph_deadline = min(deadline, time.time() + self.config.graph_time_budget)
            candidates = list(self.interesting_thm)
            random.shuffle(candidates)
            for theorem_node_id in candidates[: self.config.max_attempts]:
                if time.time() > graph_deadline:
                    break
                built = self._validated_split_from_proof(theorem_node_id)
                if built is None:
                    continue
                background, core_clauses, theorem, neg_theorem = built

                clauses = self._add_signature_distractors(background, core_clauses, theorem, neg_theorem)
                if len(clauses) < 3 or len(clauses) > self.config.max_axioms:
                    continue
                if not valid_clause_list(background + clauses + [neg_theorem]):
                    continue
                if sum(map(len, background + clauses + [neg_theorem])) > self.config.max_payload_chars:
                    continue
                quick_limit = self.config.quick_sat_time_limit
                if self._sat([neg_theorem], quick_limit, background=background) is not True:
                    continue
                if self._sat(clauses, quick_limit, background=background) is not True:
                    continue
                if any(
                    self._sat(
                        [c],
                        quick_limit,
                        background=background,
                    ) is not True
                    for c in clauses
                ):
                    continue

                random.shuffle(clauses)
                if self._sat(
                    clauses + [neg_theorem],
                    quick_limit,
                    background=background,
                ) is not False:
                    continue

                repairs = self._repair_indices(background, clauses, neg_theorem, quick_limit)
                if not repairs or len(repairs) == len(clauses):
                    continue
                if not self._final_recheck_repair(background, clauses, neg_theorem, repairs):
                    continue

                metadata = edict({
                    "background": background,
                    "clauses": clauses,
                    "theorem_node": theorem_node_id,
                    "used_split_fallback": True,
                    "split_strategy": "random_validated_useful_leaves",
                    "repair_indices": repairs,
                    "negated_theorem": neg_theorem,
                    "hypotheses": core_clauses,
                    "axiom_set": self.axiom_set,
                    "proof_depth": self.config.proof_depth,
                })
                return Entry(metadata, self._format_indices(repairs))
            self._initialize_graph()

        raise RuntimeError("failed to build a compact real TPTP consistency-repair task")

    def generate_entry(self):
        return self._build_problem()

    def render_prompt(self, metadata):
        background = "\n".join(f"- {c}" for c in metadata.get("background", []))
        clauses = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(metadata["clauses"]))

        return (
            "Which local single-clause deletions make the fixed axioms satisfiable with the negated theorem?\n"
            f"Answer with ordered, space-separated clause numbers.\n"
            f"Background axioms:\n{background}\n"
            f"Negated theorem: `{metadata['negated_theorem']}`\n"
            f"Clauses:\n{clauses}"
        )

    def _format_indices(self, xs):
        return str(list(xs)) if self.config.answer_style == "list" else " ".join(map(str, xs))

    @staticmethod
    def _parse_indices(text):
        text = str(text).strip()
        if text.startswith("["):
            try:
                xs = ast.literal_eval(text)
                return [int(x) for x in xs]
            except Exception:
                return None
        try:
            return [int(part) for part in text.split()]
        except ValueError:
            return None

    def score_answer(self, answer, entry):
        parse_indices = TPTPConsistencyRepair._parse_indices
        return float(parse_indices(answer) == parse_indices(entry.answer))
