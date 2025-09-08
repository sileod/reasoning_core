
from unified_planning.shortcuts import *
import unified_planning
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.engines import PlanGenerationResult
import glob
from pyparsing import ParseException
from tqdm.auto import tqdm; tqdm.pandas()
import timeout_decorator
import random
import re
import math
import requests
import pandas as pd
import itertools
import funcy as fc
import multiprocess as mp
from functools import wraps
import itertools
import xpflow
import glob
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
import queue
from reasoning_core import template
from reasoning_core.template import Task, Problem, Reward, register_dataset, Config
from dataclasses import dataclass, field
from collections import namedtuple
#unified_planning.shortcuts.get_environment()
from unified_planning.exceptions import UPException

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
    return list(chain.from_iterable(permutations(lst, r) for r in range(1, len(lst) + 1)))

def trivial(problem):
    goals= problem.goals[0]
    init = [k for k,v in problem.initial_values.items() if v.is_true()]
    return all(g in init for g in goals.args)

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
def generate_domain(N=5):

    problem = unified_planning.model.Problem(f"omniplan--N{N}-{time.time()}")

    # types 🧮
    ntypes = random.choice([*[1]*9,random.randint(1,N//2+1)])
    types = [f'type_{i}' for i in range(ntypes)]

    # CHANGED FROM types to user_types
    problem.types = [UserType(t) for t in types]  
    rtype = lambda: choice(problem.types)
    rr = lambda n: range(random.randint(1, n))

    problem.default = default = choice([None,None,True,False])

    problem.fluent_max_arity = fluent_max_arity = choice([2]*5+[3,4])

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
        expressions = valid_expressions(action)
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

    # Set initial state🌱
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
    for _ in rr(max(1,N//2)):
        fluent = random.choice(problem.fluents)
        objects = [random.choice(list(problem.objects(fluent.signature[i].type))) for i in range(fluent.arity)]
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
    description = []

    write_default = random.random()<write_default
    # Introduction
    description.append(f"I am playing with a set of objects.")

    # Actions
    description.append("\nHere are the actions I can do:")
    for action in problem.actions:
        params = ", ".join([f"{p.name}" for p in action.parameters])
        description.append(f"{action.name} with {params}")

    description.append("\nI have the following restrictions on my actions:")
    for action in problem.actions:
        description.append('')
        params = {p.name: p.type.name for p in action.parameters}

        if action.preconditions:
            precond_str = ", ".join([str(precond).format(**params) for precond in action.preconditions])
            description.append(f"To perform {action.name} action, the following facts need to be true: {precond_str}.")

        positive_effects = [effect for effect in action.effects if effect.value.is_true()]
        if positive_effects:
            effect_str = ", ".join([str(effect.fluent).format(**params) for effect in positive_effects])
            description.append(f"Once {action.name} action is performed the following facts will be true: {effect_str}.")

        negative_effects = [effect for effect in action.effects if not effect.value.is_true()]
        if negative_effects:
            effect_str = ", ".join([str(effect.fluent).format(**params) for effect in negative_effects])
            description.append(f"Once {action.name} action is performed the following facts will be false: {effect_str}.")

    # Objects
    objects = list(itertools.chain(*[problem.objects(t) for t in problem.user_types]))
    if objects:
        obj_by_type = {}
        for obj in objects:
            if obj.type not in obj_by_type:
                obj_by_type[obj.type] = []
            obj_by_type[obj.type].append(obj.name)
        object_description = []
        for type, objs in obj_by_type.items():
            obj_list = ", ".join([f"{type.name} {obj}" for obj in objs])
            object_description.append(obj_list)
        object_str = ", ".join(object_description)
        #description.append(f"The problem involves the following objects: {object_str}.")

    # Initial state
    initial_conditions = []
    default = pd.Series(list(problem.initial_values.values())).value_counts().index[0]

    for fluent, value in problem.initial_values.items():
        #if write_default and value==default:
        #    continue
        #initial_conditions.append(f"{fluent} is {value}")
        if value.is_true():
            initial_conditions.append(f"{fluent}")

    initial_str = ", ".join(initial_conditions)
    if write_default or not initial_str:
        description.append(f"\nEverything unspecified is {default} by default")
    if initial_str:
        description.append(f"[STATEMENT]\n As initial conditions I have that, {initial_str}.")

    # Goals
    if problem.goals:
        goal_conditions = []
        for goal in problem.goals:
            goal_conditions.append(str(goal))
        goal_str = ", ".join(goal_conditions)
        description.append(f"\nMy goal is to have that {goal_str}.")

    description = "\n".join(description)
    if not re.search(r'_type_(?!0)\d+', description):
        description=description.replace('_type_0','')
    return description

_reuse = mp.Manager().Queue()

@dataclass
class PlanningConfig(Config):
    N: int = 5
    min_na: int = 1
    max_na: int = 3

    #planner:str="fast-downward-opt"
    planner:str="pyperplan-opt"

    def update(self, c):
        self.N += c
        self.min_na += c
        self.max_na += c


class Planning(template.Task):
    def __init__(self, config=PlanningConfig()):
        super().__init__(config=config)

    def generate(self, config=PlanningConfig()):
        meta=edict()
        shutup()
        N = random.randint(4, config.N)

        while True:
            domain = generate_domain(N)
            problem = generate_problem(N, domain=domain)
            try:
                solution = solve(problem, planner=config.planner)
            except Exception as e:
                print(f"ERR: {e}")
                continue
            plan = str(solution.plan).replace('SequentialPlan:\n', '').replace('\t', '')
            meta.na = na = plan.count('(')

            if na < random.choice(list(range(config.min_na, config.max_na + 1))):
                continue

            meta.problem_english = translate(problem)
            writer = PDDLWriter(problem)
            meta.problem_pddl = writer.get_problem()
            meta.domain_pddl = writer.get_domain()
            return template.Problem(meta, plan)


    def prompt(self, meta):
        s = meta.problem_english.strip()
        s = (
            f"{s}\n"
            f"Hint: Reference solution has {meta.na} actions (may not be optimal). "
            f"Return only the plan:\n"
            f"Multiple lines if needed, one action i.e. actionx(objectx, objectx...) per line."
        )
        return s

    def score_answer(self, answer, entry):
        meta = entry['metadata']
        plan_str=to_pddl(str(answer).strip())
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
