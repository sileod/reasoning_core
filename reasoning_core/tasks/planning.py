
from unified_planning.shortcuts import BoolType, CompilationKind, Compiler, InstantaneousAction, Not, Object, OneshotPlanner, OptimalityGuarantee, PlanValidator, UserType, get_environment
import unified_planning
import unified_planning as up
from unified_planning.exceptions import UPException
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.engines import PlanGenerationResult
from pyparsing import ParseException
import timeout_decorator
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
from timeout_decorator.timeout_decorator import TimeoutError
import warnings
from easydict import EasyDict as edict
from random import choice
from unified_planning.interop import convert_problem_to_tarski
from unified_planning.interop import convert_problem_from_tarski
from dataclasses import dataclass, field
from collections import Counter, namedtuple
from reasoning_core.template import Task, Problem, Reward, Config
import logging
logging.getLogger().setLevel(logging.WARNING)
from unified_planning.shortcuts import SequentialSimulator
from unified_planning.plans import ActionInstance

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


def combinations(lst):
    return list(chain.from_iterable(permutations(lst, r) for r in range(0, len(lst) + 1)))

def trivial(problem):
    goals= problem.goals[0]
    init = [k for k,v in problem.initial_values.items() if v.is_true()]
    return all(g in init for g in goals.args)

def make_cot(problem, plan):
    simulator = SequentialSimulator(problem)
    state = simulator.get_initial_state()
    
    # Helper to clean PDDL strings
    fmt = lambda s: str(s).replace('(', ' ').replace(')', '').replace(',', '')
    
    trace = []
    goals = [fmt(g) for g in problem.goals]
    trace.append(f"Target Goals: {', '.join(goals)}")
    
    # Use _values to access state efficiently
    get_state_dict = lambda s: s._values if hasattr(s, '_values') else s.values
    
    current_facts = {fmt(k) for k, v in get_state_dict(state).items() if v.is_true()}
    
    for i, action_instance in enumerate(plan.actions):
        trace.append(f"\nStep {i+1}:")
        
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
        trace.append("  - Preconditions met. Applying action.")
        
        # State Transition
        next_state = simulator.apply(state, valid_instance)
        if next_state is None:
            trace.append("  - Error: Simulation failed.")
            break

        # Calculate Effects
        next_facts = {fmt(k) for k, v in get_state_dict(next_state).items() if v.is_true()}
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
        remaining_goals = [g for g in goals if g not in current_facts]
        if not remaining_goals and i == len(plan.actions) - 1:
            trace.append("  - Goal condition satisfied.")
        elif remaining_goals:
            trace.append(f"  - Remaining goals: {len(remaining_goals)}")

    trace.append("\nPlan found.")
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
        arity = random.randint(1, 2)
        types = random.choice([
            [rtype() for j in range(arity)],
            [rtype()]*arity])

        action = InstantaneousAction(f"action_{ai}", **{f"action_{ai}_parameter{j}_{types[j].name}": types[j] for j in range(arity)})
        for _,exp in zip(rr(N), valid_expressions(action)):

            bit=random.choice([0,1])
            if random.random()<0.8: #allow 20% effect-only
                action.add_precondition([Not(exp),exp][bit])
            if random.random()<0.1:
                bit=choice([0,1]) # noise
            action.add_effect(exp, [True, False][bit])

        problem.add_action(action)

    problem.domain_reuses=0
    return problem

def generate_problem(N=5, domain=None):
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


def compile(problem):
    with Compiler(
        problem_kind = problem.kind,
        compilation_kind = CompilationKind.NEGATIVE_CONDITIONS_REMOVING) as fixer:
        qr_result = fixer.compile(
            problem,
            CompilationKind.NEGATIVE_CONDITIONS_REMOVING
        )
        return qr_result.problem


@timeout_decorator.timeout(10)
def solve(problem, planner="pyperplan-opt", lexicographic=True):
    if "pyperplan" in planner:
        problem=compile(problem)    
    if lexicographic:
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

def translate(problem: Problem, write_default=0.5) -> str:
    desc = []
    
    # 1. Analyze Types
    # If >1 type exists, types are crucial. If 1 type, they are noise.
    types = list(problem.user_types)
    multi_type = len(types) > 1

    # --- [OBJECTS] ---
    desc.append("[OBJECTS]")
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
    desc.append("\n[ACTIONS]")
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
    desc.append("\n[STATE]")
    
    # Handle Default Value logic
    if random.random() < write_default:
        # Calculate majority value in initial state (True or False)
        vals = [v.is_true() for v in problem.initial_values.values()]
        # If vals is empty, default to False
        default_val = Counter(vals).most_common(1)[0][0] if vals else False
        desc.append(f"Default: {default_val}")

    # Init (Standard PDDL assumption: list only True facts)
    init = [str(f) for f, v in problem.initial_values.items() if v.is_true()]
    desc.append(f"Initial true values: {', '.join(init) if init else 'None'}")

    desc.append('\n[GOAL]\n')
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
    arity_weight = 0.5
    hint_proba = 0.5
    #planner:str="fast-downward-opt"
    planner:str="pyperplan-opt"
    language: str = "en"
    domain: str = None
    #domains: list = field(default_factory=lambda: ["blocksworld", "mystery", None])
    domains: list = field(default_factory=lambda: [None])
    def update(self, c):
        self.N += c
        self.min_na += c
        self.max_na += c
        self.arity_weight += c

class Planning(Task):
    task_name = "planning" 

    def __init__(self, config=PlanningConfig()):
        super().__init__(config=config)
        shutup()

    def generate(self):
        meta=edict()
        config = self.config
        config.domain = random.choice(config.domains)
        N = random.randint(4, config.N)

        while True:
    
            meta.domain_seed = f"{N}-{random.randint(0,config.max_domain_seed)}"
            meta.fluent_arity = fma = random.choices([1, 2], weights=[1, config.arity_weight], k=1)[0]

            domain = generate_domain(N, meta.domain_seed, fluent_max_arity=fma) if not config.domain else fetch_domain(config.domain)
            random.seed(None)
            problem = generate_problem(N, domain=domain)
            try:
                solution = solve(problem, planner=config.planner)
            except Exception as e:
                print(f"ERR: {e}")
                continue
            plan = str(solution.plan).replace('SequentialPlan:\n', '').replace('\t', '')

            meta.na = na = plan.count('(')

            if na < random.choice(list(range(config.min_na, config.max_na + 1))):
                continue # ensure plan is long enough

            meta.problem_english = translate(problem)
            writer = PDDLWriter(problem)
            meta.problem_pddl = writer.get_problem()
            meta.domain_pddl = writer.get_domain()
            meta.cot = make_cot(problem, solution.plan)
            if self.score_answer(plan, {'metadata': meta})<1:
                continue
            return Problem(meta, plan)


    def prompt(self, meta):
        txt = meta.problem_english.strip()
        txt += "\n\n[OUTPUT]"
        
        if random.random() < self.config.hint_proba:
            txt += f"\nHint: Reference solution has {meta.na} actions (but it may not be optimal)."
        txt += (
            "\nReturn only the plan."
            "\nFormat: Multiple lines, one action per line: action(obj1, obj2)"
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
