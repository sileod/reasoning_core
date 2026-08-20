
from unified_planning.shortcuts import BoolType, CompilationKind, Compiler, InstantaneousAction, Not, Object, OneshotPlanner, OptimalityGuarantee, PlanValidator, UserType, get_environment
import unified_planning
import unified_planning as up
from unified_planning.exceptions import UPException
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.engines import PlanGenerationResult
from pyparsing import ParseException
import random
import re
import math
import requests
import pandas as pd
import itertools
from functools import wraps
import itertools
import json
from itertools import permutations, chain
import time
from functools import wraps
from traceback import format_exc
import warnings
from easydict import EasyDict as edict
from random import choice
from unified_planning.interop import convert_problem_to_tarski
from unified_planning.interop import convert_problem_from_tarski
from dataclasses import dataclass, field
from collections import Counter, namedtuple
from reasoning_core.template import Task, Entry, Reward, Config
import logging
logging.getLogger().setLevel(logging.WARNING)
from unified_planning.shortcuts import SequentialSimulator
from unified_planning.plans import ActionInstance, SequentialPlan

Range = namedtuple('Range', 'low high type')

def backtr(x):
    for _ in range(100):
        try:
            tarski_problem = convert_problem_to_tarski(x)
            problem = convert_problem_from_tarski(get_environment(), tarski_problem)
            return problem
        except Exception as e:
            pass
    print('ERR')

def shutup():
    unified_planning.shortcuts.get_environment().credits_stream=None
    warnings.filterwarnings("ignore", message=".*not support custom heuristic*")
    warnings.filterwarnings("ignore", message=".*cannot establish whether*")
    warnings.filterwarnings("ignore", message=".*does not support timeout")
    logging.disable(logging.INFO) 


def combinations(lst):
    return list(chain.from_iterable(permutations(lst, r) for r in range(0, len(lst) + 1)))

def trivial(problem):
    goals= problem.goals[0]
    init = [k for k,v in problem.initial_values.items() if v.is_true()]
    return all(g in init for g in goals.args)

def ground_fluent_expressions(problem):
    for fluent in problem.fluents:
        obj_lists = [
            list(problem.objects(param.type))
            for param in fluent.signature
        ]
        for args in itertools.product(*obj_lists) if obj_lists else [()]:
            yield fluent(*args)

def true_facts(problem, state):
    fmt = lambda s: str(s).replace('(', ' ').replace(')', '').replace(',', '')
    return {
        fmt(atom)
        for atom in ground_fluent_expressions(problem)
        if state.get_value(atom).is_true()
    }

def goal_satisfied(goal, state):
    if goal.is_not():
        return state.get_value(goal.arg(0)).is_false()
    return state.get_value(goal).is_true()

def fact_key(fact):
    return str(fact)

def literal_key(expr):
    if expr.is_not():
        return (fact_key(expr.arg(0)), False)
    return (fact_key(expr), True)

def state_fact_nodes(problem, state):
    facts = {}
    for atom in ground_fluent_expressions(problem):
        if state.get_value(atom).is_true():
            facts[fact_key(atom)] = atom
    return facts

def format_plan(plan):
    return "\n".join(str(a) for a in plan.actions)

def rebind_plan(problem, plan):
    em = problem.environment.expression_manager
    actions = []
    for action_instance in plan.actions:
        action = problem.action(action_instance.action.name)
        params = []
        for p in action_instance.actual_parameters:
            if p.is_object_exp():
                params.append(em.ObjectExp(problem.object(p.object().name)))
            else:
                params.append(em.ObjectExp(problem.object(str(p))))
        actions.append(ActionInstance(action, tuple(params)))
    return SequentialPlan(actions)

def simulate_plan_valid(problem, plan):
    simulator = SequentialSimulator(problem)
    state = simulator.get_initial_state()
    for action_instance in plan.actions:
        if not simulator.is_applicable(state, action_instance):
            return False
        state = simulator.apply(state, action_instance)
        if state is None:
            return False
    return all(goal_satisfied(goal, state) for goal in problem.goals)

def make_cot(problem, plan):
    simulator = SequentialSimulator(problem)
    state = simulator.get_initial_state()
    
    # Helper to clean PDDL strings
    fmt = lambda s: str(s).replace('(', ' ').replace(')', '').replace(',', '')
    
    trace = []
    goals = [fmt(g) for g in problem.goals]
    trace.append(f"Target Goals: {', '.join(goals)}")
    
    current_facts = true_facts(problem, state)

    for i, action_instance in enumerate(plan.actions):        
        # --- RE-BINDING LOGIC ---
        # 1. Get action schema from original problem
        act_name = action_instance.action.name
        original_action = problem.action(act_name)
        
        # 2. Map parameters from Plan (FNodes) -> Original Problem (Objects)
        original_params = []
        for p in action_instance.actual_parameters:
            if p.is_object_exp():
                # Extract the UP Object from the FNode, then get its name
                obj_name = p.object().name
            else:
                # Fallback for constants (e.g. Bool/Int)
                obj_name = str(p)
            
            # Fetch the specific object instance from the original problem
            original_params.append(problem.object(obj_name))
            
        # 3. Create valid instance for this simulator
        valid_instance = ActionInstance(original_action, tuple(original_params))
        # ------------------------

        # Formatting
        params_str = [str(p) for p in valid_instance.actual_parameters]
        action_str = f"({valid_instance.action.name} {' '.join(params_str)})"
        
        # Verify Preconditions
        if not simulator.is_applicable(state, valid_instance):
            trace.append(f"ERR: Action {action_str} is not applicable in current state.")
            break
            
        trace.append(f"Selected Action: {action_str}")
        
        # State Transition
        next_state = simulator.apply(state, valid_instance)
        if next_state is None:
            trace.append("  - Error: Simulation failed.")
            break

        # Calculate Effects
        next_facts = true_facts(problem, next_state)
        added = next_facts - current_facts
        removed = current_facts - next_facts
        
        if added:
            trace.append(f"  - Added effects: {', '.join(sorted(list(added)))}")
        if removed:
            trace.append(f"  - Removed effects: {', '.join(sorted(list(removed)))}")
            
        # Update loop state
        current_facts = next_facts
        state = next_state
        
        # Goal check
        remaining_goals = [g for g in problem.goals if not goal_satisfied(g, state)]
        if not remaining_goals and i == len(plan.actions) - 1:
            trace.append("  - Goal condition satisfied.")
        elif remaining_goals:
            trace.append(f"  - Remaining goals: {len(remaining_goals)}")

    return "\n".join(trace)

def fetch_domain(domain):
    
    base_url = "https://raw.githubusercontent.com/karthikv792/LLMs-Planning/main/plan-bench/instances/blocksworld"
    domain_url = {
        "generated_basic": f"{base_url}/generated_domain.pddl",
        "mystery": f"{base_url}/mystery/generated_domain.pddl"
    }
    domain_url["blocksworld"] = domain_url["generated_basic"]
    assert domain in domain_url
    
    get_domain = lambda cfg: requests.get(domain_url[cfg]).text
    return PDDLReader().parse_problem_string(get_domain(domain))


def rolling(n_times):
    def decorator(func):
        cache = []  # Store cached results
        call_count = [0]  # Track how many times the last cached value was returned

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we need a new value (if cache is empty or last value has been used n_times)
            if not cache or call_count[0] >= n_times:
                cache.append(func(*args, **kwargs))
                if len(cache) > n_times:  # Limit cache size to n_times
                    cache.pop(0)
                call_count[0] = 1
            else:
                call_count[0] += 1  # Increment the use counter for the last cached value

            return cache[-1]

        return wrapper
    return decorator


#@rolling(10)
def generate_domain(N=5, seed=None, fluent_max_arity=2):
    state = random.getstate()
    random.seed(seed)
    problem = unified_planning.model.Problem(f"omniplan--N{N}-seed{seed}")

    # types 🧮
    ntypes = random.choice([*[1]*9,random.randint(1,N//2+1)])
    types = [f'type_{i}' for i in range(ntypes)]

    # CHANGED FROM types to user_types
    problem.types = [UserType(t) for t in types]  
    rtype = lambda: choice(problem.types)
    rr = lambda n: range(random.randint(1, n))

    problem.default = default = choice([None,None,True,False])

    problem.fluent_max_arity = fluent_max_arity

    # Generate ~N fluents 🏷️
    for i in rr(N):
        arity = random.randint(0, fluent_max_arity)  # Allow for fluents with 0, 1 or 2 parameters

        types = random.choice([
            [rtype() for j in range(arity)],
            [rtype()]*arity])
        problem.add_fluent(
            f"fluent_{i}",
            BoolType(),
            **{f"parameter{j}": types[j] for j in range(arity )},
            default_initial_value=default
        )

    def valid_expressions(action):
        parameters_combinations = combinations(action.parameters)
        types = lambda x: [a.type for a in x]

        exp=[]
        for f in problem.fluents:
            exp+=[f(*pc) for pc in parameters_combinations if types(pc)==types(f.signature) ]
        random.shuffle(exp)
        return exp

    #problem.add_action(InstantaneousAction('null'))

    # Generate ~N actions 🔨
    for ai in rr(N):
        for _ in range(12):
            arity = random.randint(1, 2)
            types = random.choice([
                [rtype() for j in range(arity)],
                [rtype()]*arity])

            action = InstantaneousAction(f"action_{ai}", **{f"action_{ai}_parameter{j}_{types[j].name}": types[j] for j in range(arity)})
            expressions = valid_expressions(action)
            if expressions:
                break
        else:
            continue

        n_pre = random.randint(0, min(len(expressions), max(1, N // 2)))
        pre_exps = random.sample(expressions, k=n_pre)
        pre_literals = {}
        for exp in pre_exps:
            pre_literals[fact_key(exp)] = random.choice([True, False])
            action.add_precondition(exp if pre_literals[fact_key(exp)] else Not(exp))

        effect_pool = expressions[:]
        random.shuffle(effect_pool)
        n_eff = random.randint(1, min(len(effect_pool), max(1, N // 2)))
        effects = {}
        for exp in effect_pool:
            key = fact_key(exp)
            if key in effects:
                continue
            value = random.choice([True, False])
            if pre_literals.get(key) == value and len(effect_pool) > 1:
                value = not value
            effects[key] = (exp, value)
            if len(effects) >= n_eff:
                break

        for exp, value in effects.values():
            action.add_effect(exp, value)

        problem.add_action(action)

    problem.domain_reuses=0
    random.setstate(state)
    return problem

def generate_problem(N=5, domain=None, add_random_goals=True):
    rr = lambda n: range(random.randint(1, n))

    if not domain:
        problem = generate_domain(N=N)
    else:
        problem=domain.clone()
        problem.fluent_max_arity=2

    init_rate = random.random()**2.5
    if problem.fluent_max_arity>2:
        init_rate**=problem.fluent_max_arity

    problem = problem.clone()

    # Generate objects 🧱
    i=0
    for t in problem.user_types:
        for _ in rr(N):
            i+=1
            if len(problem.user_types)==1:
                type_suffix = ''
            else:
                type_suffix = f"_{t.name}"
            obj = Object(f"object_{i}{type_suffix}", t)
            problem.add_object(obj)

    # Set initial state 🌱
    init = lambda: random.random()<init_rate

    for fluent in problem.fluents:
        object_combinations = itertools.product(*[
            list(problem.objects(fluent.signature[i].type))
            for i in range(fluent.arity)
        ])
        if fluent.arity==0:
            object_combinations = [[]]
        for objects in object_combinations:
            value = init()
            #if value==problem.default:
            #    continue
            problem.set_initial_value(fluent(*objects), value)


    # Set goal state 🏁
    if add_random_goals:
        rr = lambda n: range(random.randint(1, n))
        used_goals = set()

        for _ in rr(max(1,N//2)):
            fluent = random.choice(problem.fluents)
            objects = [random.choice(list(problem.objects(fluent.signature[i].type))) for i in range(fluent.arity)]
            objects = tuple(objects)
            if (fluent, objects) in used_goals:
                continue
            used_goals.add((fluent, objects))

            expr = fluent(*objects)
            expr = random.choice([Not(expr)]+5*[expr])
            problem.add_goal(expr)
    problem.domain=domain
    return problem

def ground_action(problem, simulator, action, params):
    em = problem.environment.expression_manager
    fnode_params = tuple(p if hasattr(p, "is_object_exp") else em.ObjectExp(p) for p in params)
    return simulator._ground_action(action, fnode_params)

def ground_effect_literals(problem, simulator, action, params):
    grounded = ground_action(problem, simulator, action, params)
    effects = []
    for effect in grounded.effects:
        if effect.is_conditional() or effect.is_forall():
            continue
        effects.append((fact_key(effect.fluent), effect.fluent, effect.value.is_true()))
    return effects

def positive_achiever_counts(problem):
    simulator = SequentialSimulator(problem)
    counts = Counter()
    for action in problem.actions:
        obj_lists = [list(problem.objects(param.type)) for param in action.parameters]
        for params in itertools.product(*obj_lists) if obj_lists else [()]:
            try:
                for key, _, value in ground_effect_literals(problem, simulator, action, params):
                    if value:
                        counts[key] += 1
            except Exception:
                continue
    return counts

def sample_weighted(items, scores, temperature=2.0):
    m = max(scores)
    weights = [math.exp((s - m) / temperature) for s in scores]
    total = sum(weights)
    pick = random.random() * total
    acc = 0
    for item, weight in zip(items, weights):
        acc += weight
        if acc >= pick:
            return item
    return items[-1]

def applicable_action_instance(applicable):
    if isinstance(applicable, ActionInstance):
        return applicable
    action, params = applicable
    return ActionInstance(action, tuple(params))

def novelty_walk(problem, length, temperature=2.0):
    simulator = SequentialSimulator(problem)
    state = simulator.get_initial_state()
    states = [state]
    plan = []
    current_true = set(state_fact_nodes(problem, state))
    seen_states = {frozenset(current_true)}
    prev_added = set()
    prev_deleted = set()

    for _ in range(length):
        scored = []
        for applicable in simulator.get_applicable_actions(state):
            ai = applicable_action_instance(applicable)
            try:
                effects = ground_effect_literals(problem, simulator, ai.action, ai.actual_parameters)
            except Exception:
                continue
            adds = {key for key, _, value in effects if value and key not in current_true}
            deletes = {key for key, _, value in effects if not value and key in current_true}
            next_true = (current_true | adds) - deletes
            score = 3 * len(adds) + len(deletes)
            if frozenset(next_true) in seen_states:
                score -= 4
            if adds & prev_deleted or deletes & prev_added:
                score -= 3
            if not adds and not deletes:
                score -= 2
            scored.append(((ai, adds, deletes), score))

        if not scored:
            break

        instance, added, deleted = sample_weighted(
            [x for x, _ in scored],
            [s for _, s in scored],
            temperature=temperature,
        )
        next_state = simulator.apply(state, instance)
        if next_state is None:
            break
        plan.append(instance)
        state = next_state
        states.append(state)
        current_true = set(state_fact_nodes(problem, state))
        seen_states.add(frozenset(current_true))
        prev_added = added
        prev_deleted = deleted

    return SequentialPlan(plan), states

def choose_late_goals(problem, states, max_goals=1, achievers=None):
    if len(states) < 2:
        return []
    init_true = set(state_fact_nodes(problem, states[0]))
    final_nodes = state_fact_nodes(problem, states[-1])
    final_true = set(final_nodes)
    changed = final_true - init_true
    if not changed:
        return []

    first_seen = {}
    for i, state in enumerate(states[1:], start=1):
        for key in set(state_fact_nodes(problem, state)) - init_true:
            first_seen.setdefault(key, i)

    if achievers is None:
        achievers = positive_achiever_counts(problem)
    candidates = [key for key in changed if first_seen.get(key, 0) >= max(1, len(states) // 2)]
    for upper in (2, 4, 10**9):
        filtered = [key for key in candidates if 1 <= achievers.get(key, 0) <= upper]
        if filtered:
            candidates = filtered
            break
    if not candidates:
        candidates = list(changed)

    candidates.sort(key=lambda key: (first_seen.get(key, 0), -achievers.get(key, 0)), reverse=True)
    top = candidates[:max(4, max_goals * 3)]
    random.shuffle(top)
    return [final_nodes[key] for key in top[:max_goals]]

def prune_plan_to_goals(problem, raw_plan, goals):
    simulator = SequentialSimulator(problem)
    required = {literal_key(goal) for goal in goals}
    kept = []

    for action_instance in reversed(raw_plan.actions):
        params = tuple(action_instance.actual_parameters)
        effects = ground_effect_literals(problem, simulator, action_instance.action, params)
        effect_literals = {(key, value) for key, _, value in effects}
        if not (required & effect_literals):
            continue

        kept.append(action_instance)
        for key, _, value in effects:
            required.discard((key, value))
        grounded = ground_action(problem, simulator, action_instance.action, params)
        for precondition in grounded.preconditions:
            if precondition.is_not():
                required.add((fact_key(precondition.arg(0)), False))
            else:
                required.add((fact_key(precondition), True))

    kept.reverse()
    return SequentialPlan(kept)

def visible_fact_keys(problem, plan):
    simulator = SequentialSimulator(problem)
    keys = {fact_key(goal.arg(0) if goal.is_not() else goal) for goal in problem.goals}
    for action_instance in plan.actions:
        grounded = ground_action(problem, simulator, action_instance.action, tuple(action_instance.actual_parameters))
        for precondition in grounded.preconditions:
            keys.add(fact_key(precondition.arg(0) if precondition.is_not() else precondition))
        for effect in grounded.effects:
            keys.add(fact_key(effect.fluent))
    return keys

def trim_problem(problem, plan, level=0):
    modes = ["plan_cone", "medium", "full"]
    weights_by_level = [
        [0.70, 0.25, 0.05],
        [0.45, 0.40, 0.15],
        [0.25, 0.50, 0.25],
        [0.10, 0.55, 0.35],
    ]
    weights = weights_by_level[min(level, len(weights_by_level) - 1)]
    mode = random.choices(modes, weights=weights, k=1)[0]
    trimmed = problem.clone()
    if mode == "full":
        return trimmed, mode

    plan_action_names = {a.action.name for a in plan.actions}
    selected = [a for a in trimmed.actions if a.name in plan_action_names]
    distractor_rate = 0.20 + 0.12 * min(level, 5)
    if mode == "medium":
        for action in trimmed.actions:
            if action.name not in plan_action_names and random.random() < distractor_rate:
                selected.append(action)

    trimmed.clear_actions()
    for action in selected:
        trimmed.add_action(action.clone())

    keep_facts = visible_fact_keys(problem, plan)
    if mode == "medium":
        for fact in list(problem.initial_values):
            if fact_key(fact) not in keep_facts and random.random() < distractor_rate:
                keep_facts.add(fact_key(fact))

    for fact, value in list(trimmed.initial_values.items()):
        if value.is_true() and fact_key(fact) not in keep_facts:
            trimmed.set_initial_value(fact, False)
    return trimmed, mode

def generate_planted_problem(N, domain, target_len, level=0, max_attempts=80):
    for _ in range(max_attempts):
        problem = generate_problem(N, domain=domain, add_random_goals=False)
        walk_len = random.randint(target_len + 1, target_len + 5 + max(0, level))
        raw_plan, states = novelty_walk(problem, walk_len)
        if len(raw_plan.actions) < target_len:
            continue
        achievers = positive_achiever_counts(problem)
        goal_cap = 1 + int(level >= 1) + int(target_len >= 4) + max(0, level // 2)
        goal_count = random.randint(1, min(5, goal_cap))
        goals = choose_late_goals(problem, states, max_goals=goal_count, achievers=achievers)
        if not goals:
            continue
        problem.clear_goals()
        for goal in goals:
            problem.add_goal(goal)
        plan = prune_plan_to_goals(problem, raw_plan, goals)
        if len(plan.actions) < target_len:
            continue
        if not simulate_plan_valid(problem, plan):
            continue
        trimmed, trim_mode = trim_problem(problem, plan, level=level)
        trimmed_plan = rebind_plan(trimmed, plan)
        if simulate_plan_valid(trimmed, trimmed_plan):
            return trimmed, trimmed_plan, trim_mode
    raise RuntimeError("Could not generate a planted planning problem")


def compile(problem):
    with Compiler(
        problem_kind = problem.kind,
        compilation_kind = CompilationKind.NEGATIVE_CONDITIONS_REMOVING) as fixer:
        qr_result = fixer.compile(
            problem,
            CompilationKind.NEGATIVE_CONDITIONS_REMOVING
        )
        return qr_result.problem


#@timeout_decorator.timeout(10)
def solve(problem, planner="pyperplan-opt", rank_tiebreak=True):
    if "pyperplan" in planner:
        problem=compile(problem)    
    if rank_tiebreak:
        costs = {a: 10000+i for i,a in enumerate(problem.actions)}
        problem.add_quality_metric(up.model.metrics.MinimizeActionCosts(costs))

    og = OptimalityGuarantee.SOLVED_OPTIMALLY
    try:
        with OneshotPlanner(name=planner,
            problem_kind=problem.kind, optimality_guarantee=og) as planner:   
            result = planner.solve(problem,timeout=8)
    except TimeoutError:
        return PlanGenerationResult("ERR:timeout",[],planner)
    return result



def to_pddl(s):
    actions = [a.strip('[]').strip().replace(',','').replace('(',' ') for a in s.split(')')]
    return "\n".join([f'({a})' for a in actions if a]).replace('))',')')

def translate(problem) -> str:
    desc = []
    
    # 1. Analyze Types
    # If >1 type exists, types are crucial. If 1 type, they are noise.
    types = list(problem.user_types)
    multi_type = len(types) > 1

    # --- [OBJECTS] ---
    desc.append("Objects:")
    if multi_type:
        for t in types:
            objs = list(problem.objects(t))
            if objs:
                desc.append(f"{t.name}: {', '.join(o.name for o in objs)}")
    else:
        # Flatten list if types don't matter
        all_objs = list(itertools.chain.from_iterable(problem.objects(t) for t in types))
        desc.append(", ".join(o.name for o in all_objs) if all_objs else "None")

    # --- [ACTIONS] ---
    desc.append("\nActions:")
    for action in problem.actions:
        # Map internal names (action_0_param_1) to logical names (x0, x1)
        # We sort by length descending so regex doesn't match substrings (e.g. param1 inside param10)
        param_map = {p.name: f"x{i}" for i, p in enumerate(action.parameters)}
        sorted_keys = sorted(param_map.keys(), key=len, reverse=True)

        def clean(expr):
            s = str(expr)
            for old in sorted_keys:
                # Regex \b ensures exact word matching
                s = re.sub(rf"\b{re.escape(old)}\b", param_map[old], s)
            return s

        # Signature
        params = []
        for i, p in enumerate(action.parameters):
            p_str = f"x{i}:{p.type.name}" if multi_type else f"x{i}"
            params.append(p_str)
        desc.append(f"{action.name}({', '.join(params)})")

        # Logic
        if action.preconditions:
            pre = ", ".join([clean(p) for p in action.preconditions])
            desc.append(f"  Requires: {pre}")

        # Combine positive and negative effects into one line
        effects = []
        for e in action.effects:
            fluent = clean(e.fluent)
            if e.value.is_true():
                effects.append(fluent)
            else:
                effects.append(f"not {fluent}")
        
        if effects:
            desc.append(f"  Effect: {', '.join(effects)}")

    # --- [STATE] ---
    desc.append("\nInitial state:")

    # Init (Standard PDDL assumption: list only True facts)
    init = [str(f) for f, v in problem.initial_values.items() if v.is_true()]
    desc.append(f"True values: {', '.join(init) if init else 'None'}")
    desc.append("All facts not listed under True values are false.")

    desc.append('\nGoal:')
    goals = [str(g) for g in problem.goals]
    desc.append(', '.join(goals) if goals else 'None')
    
    return "\n".join(desc)




def parse_jsonl_plan(jsonl_string: str) -> str:
    """
    Parses a JSONL string of tool calls into a PDDL-like plan string.
    """
    actions = []
    for line in jsonl_string.strip().splitlines():
        try:
            # Parse the JSON from the line
            call = json.loads(line)
            tool_name = call.get("tool_name")
            args = call.get("arguments", {})
            
            if not tool_name or not isinstance(args, dict):
                continue # Skip malformed lines

            # Format into PDDL-like action: (action_name arg1 arg2)
            arg_values = " ".join(args.values())
            actions.append(f"({tool_name} {arg_values})".strip())
        except (json.JSONDecodeError, AttributeError):
            # Ignore lines that are not valid JSON or don't have the expected structure
            continue
            
    return "\n".join(actions)


@dataclass
class PlanningConfig(Config):
    N: int = 5
    min_na: int = 1
    max_na: int = 3
    max_domain_seed: int = 500
    arity_weight: float = 0.5
    hint_proba: float = 0.5
    pure_random_proba: float = 0.12
    optimal_relabel: bool = True
    audit_proba: float = 0.0
    #planner:str="fast-downward-opt"
    planner: str = "pyperplan-opt"
    language: str = "en"
    domain: object = None
    #domains: list = field(default_factory=lambda: ["blocksworld", "mystery", None])
    domains: list = field(default_factory=lambda: [None])
    def apply_difficulty(self, level):
        self.N += level
        self.min_na += level
        self.max_na += level
        self.arity_weight += level
        self.pure_random_proba *= max(0, 1 - level / 3)

class Planning(Task):
    summary = "Generate action plans to achieve goals in domains like Blocksworld."
    task_name = "planning" 

    def __init__(self, config=None):
        super().__init__(config=config or PlanningConfig())
        shutup()

    def generate_entry(self):
        meta=edict()
        config = self.config
        config.domain = random.choice(config.domains)
        level = getattr(config, "level", 0)
        N = random.randint(4, config.N)
        target_na = random.choice(list(range(config.min_na, config.max_na + 1)))

        for _ in range(250):
            try:
                meta.domain_seed = f"{N}-{random.randint(0,config.max_domain_seed)}"
                meta.fluent_arity = fma = random.choices([1, 2], weights=[1, config.arity_weight], k=1)[0]

                domain = generate_domain(N, meta.domain_seed, fluent_max_arity=fma) if not config.domain else fetch_domain(config.domain)
                if random.random() < config.pure_random_proba:
                    problem = generate_problem(N, domain=domain)
                    solution = solve(problem, planner=config.planner)
                    if not solution.plan:
                        continue
                    reference_plan = solution.plan
                    trim_mode = "full"
                    generator_mode = "random_solve"
                else:
                    problem, reference_plan, trim_mode = generate_planted_problem(
                        N,
                        domain,
                        target_na,
                        level=level,
                    )
                    generator_mode = "planted_walk"

                planted_na = len(reference_plan.actions)
                if generator_mode == "planted_walk" and config.optimal_relabel:
                    solution = solve(problem, planner=config.planner)
                    if solution.plan and len(solution.plan.actions) >= config.min_na:
                        reference_plan = solution.plan
                        generator_mode = "planted_walk_optimal"
                if generator_mode == "planted_walk" and len(reference_plan.actions) < target_na:
                    continue

                if generator_mode == "planted_walk" and random.random() < config.audit_proba:
                    solution = solve(problem, planner=config.planner)
                    if solution.plan and 0 < len(solution.plan.actions) >= config.min_na:
                        reference_plan = solution.plan
                        generator_mode = "planted_walk_audited"

                plan = format_plan(reference_plan)
                meta.na = len(reference_plan.actions)
                meta.planted_na = planted_na
                meta.optimality_gap = planted_na - meta.na
                meta.target_na = target_na
                meta.generator_mode = generator_mode
                meta.trim_mode = trim_mode

                meta.problem_english = translate(problem)
                writer = PDDLWriter(problem)
                meta.problem_pddl = writer.get_problem()
                meta.domain_pddl = writer.get_domain()
                meta.verif_cot = make_cot(problem, reference_plan) #deprecated cot
                if self.score_answer(plan, {'metadata': meta})<1:
                    continue
                return Entry(meta, plan)
            except Exception:
                continue
        raise RuntimeError("Could not generate a planning problem")


    def render_prompt(self, meta):
        txt = meta.problem_english.strip()       
        if random.random() < self.config.hint_proba:
            if meta.get("generator_mode") == "planted_walk_optimal":
                txt += f"\nHint: A shortest reference solution has {meta.na} actions."
            else:
                txt += f"\nHint: Reference solution has {meta.na} actions (but it may not be optimal)."
        answer_kind = "shortest valid" if meta.get("generator_mode") in {"planted_walk_optimal", "planted_walk_audited", "random_solve"} else "valid"
        txt += (
            "\n\nAction format example: action_0(object1, object2)."
            f"\nThe answer is a {answer_kind} plan, one action per line."
        )
        return txt

    def score_answer(self, answer, entry):
        meta = entry['metadata']

        answer = str(answer).strip()
        if meta.get('language')=="tool_calling":
            plan_str=parse_jsonl_plan(str(answer).strip())
        else:
            plan_str=to_pddl(answer)
    
        reader = PDDLReader()
        d,p = meta.get('domain_pddl'), meta.get('problem_pddl')
        pddl = reader.parse_problem_string(d,p)
        try:
            plan=reader.parse_plan_string(pddl, plan_str)
            assert len(plan_str.strip())
        except:
            return Reward(0, 'plan parsing error')

        with PlanValidator(name="sequential_plan_validator", problem_kind=pddl.kind, plan_kind=pddl.kind) as validator:
            if str(validator.validate(pddl, plan).status)=='ValidationResultStatus.VALID':
                return Reward(1)
            else:
                return Reward(0.1,'bad_semantics')
