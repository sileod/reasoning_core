"""Cue-unique grounded STRIPS planning."""
import math
import random
from dataclasses import dataclass
from functools import lru_cache

from reasoning_core.template import Config, Entry, Reward, Task, edict


PREDICATES = (
    "active", "aligned", "charged", "clear", "closed", "cool", "dry", "fixed",
    "free", "linked", "marked", "open", "ready", "safe", "stable", "warm",
)
OBJECTS = (
    "amber", "birch", "coral", "delta", "elm", "fjord", "grove", "harbor",
    "indigo", "jade", "kestrel", "linen",
)
VERBS = (
    "align", "bind", "carry", "close", "cool", "drain", "fasten", "guide",
    "join", "lift", "mark", "open", "prime", "release", "route", "shift",
)


def _action(call, pre_true, pre_false, add, delete):
    return edict(call=call, pre_true=sorted(pre_true), pre_false=sorted(pre_false),
                 add=sorted(set(add) - set(delete)), delete=sorted(delete))


def _applicable(action, state):
    return set(action["pre_true"]) <= state and not set(action["pre_false"]) & state


def _apply(action, state):
    return (state - set(action["delete"])) | set(action["add"])


def _phase(stage, facts, style):
    if style == "onehot":
        return {facts[stage]}, set()
    gray = stage ^ (stage >> 1)
    yes = {fact for i, fact in enumerate(facts) if gray & (1 << i)}
    return yes, set(facts) - yes


def _solutions(model, cues, cap=2):
    phases = [tuple(map(frozenset, _phase(i, model.phase_facts, model.style)))
              for i in range(model.horizon + 1)]

    def stage_of(state):
        return next((i for i, (yes, no) in enumerate(phases)
                     if yes <= state and not no & state), None)

    @lru_cache(None)
    def visit(state_key, step):
        state = set(state_key)
        stage = stage_of(state)
        if stage is None or model.horizon - step < model.horizon - stage:
            return ()
        if step == model.horizon:
            valid = set(model.goal_true) <= state and not set(model.goal_false) & state
            return ((),) if valid else ()
        out = []
        for action in model.actions:
            if cues.get(step) not in (None, action.call) or not _applicable(action, state):
                continue
            nxt = _apply(action, state)
            if nxt == state:
                continue
            for tail in visit(frozenset(nxt), step + 1):
                out.append((action.call,) + tail)
                if len(out) >= cap:
                    return tuple(out)
        return tuple(out)

    return visit(frozenset(model.initial_true), 0)


def _generate(level, horizon):
    style = random.choice(("onehot", "gray"))
    n_phase = horizon + 1 if style == "onehot" else math.ceil(math.log2(horizon + 1))
    atoms = random.sample([f"{p}({o})" for p in PREDICATES for o in OBJECTS],
                          n_phase + 7 + level)
    phase_facts, context = atoms[:n_phase], atoms[n_phase:]
    calls = iter(random.sample([f"{v}({a},{b})" for v in VERBS
                                for a in OBJECTS for b in OBJECTS if a != b], 80))
    initial_phase, _ = _phase(0, phase_facts, style)
    state = set(initial_phase) | set(random.sample(context, max(2, len(context) // 2)))
    guard = random.choice(sorted(state - set(initial_phase)))
    mutable = [fact for fact in context if fact != guard]
    initial = set(state)
    actions, spine, stage_states = [], [], []

    for i in range(horizon):
        stage_states.append(set(state))
        now_true, now_false = _phase(i, phase_facts, style)
        next_true, _ = _phase(i + 1, phase_facts, style)
        ctx_true = list((state & set(context)) - {guard})
        ctx_false = list(set(context) - state)
        pre_true = set(now_true) | {guard} | set(random.sample(ctx_true, min(len(ctx_true), level // 3)))
        pre_false = set(now_false) | set(random.sample(ctx_false, min(len(ctx_false), level // 4)))
        add, delete = next_true - now_true, now_true - next_true
        if mutable and random.random() < .65:
            fact = random.choice(mutable)
            (delete if fact in state else add).add(fact)
        action = _action(next(calls), pre_true, pre_false, add, delete)
        actions.append(action)
        spine.append(action.call)
        state = _apply(action, state)

    if level >= 3:
        count = min(max(1, horizon // 4), horizon - 2)
        for i in random.sample(range(1, horizon - 1), count):
            base = next(a for a in actions if a.call == spine[i])
            actions.append(_action(next(calls), base.pre_true, base.pre_false, base.add, base.delete))

    for _ in range(2 + level):
        i = random.randrange(horizon)
        base = next(a for a in actions if a.call == spine[i])
        if i < horizon - 1 and random.random() < .65:
            actions.append(_action(next(calls), base.pre_true, base.pre_false,
                                   base.add, set(base.delete) | {guard}))
        else:
            fact, before = random.choice(mutable), stage_states[i]
            actions.append(_action(next(calls), base.pre_true, base.pre_false,
                                   {fact} if fact not in before else set(),
                                   {fact} if fact in before else set()))

    goal_true, goal_false = _phase(horizon, phase_facts, style)
    random.shuffle(actions)
    model = edict(actions=actions, initial_true=sorted(initial), goal_true=sorted(goal_true),
                  goal_false=sorted(goal_false), horizon=horizon, phase_facts=phase_facts,
                  style=style)
    cues, reference = {}, tuple(spine)
    while True:
        solutions = _solutions(model, cues)
        if len(solutions) == 1:
            break
        if not solutions:
            raise RuntimeError("constructive planning produced no solution")
        alternative = next((x for x in solutions if x != reference), solutions[-1])
        divergence = next(i for i, pair in enumerate(zip(reference, alternative))
                          if pair[0] != pair[1])
        cues[divergence] = reference[divergence]
        if len(cues) > math.ceil(horizon / 3):
            raise RuntimeError("constructive planning needed too many cues")

    metadata = edict(
        engine="bounded-strips-v1", horizon=horizon, style=style,
        initial_true=model.initial_true, actions=actions,
        goal_true=model.goal_true, goal_false=model.goal_false,
        plan_cue=edict(length=horizon, steps=[{"step": i + 1, "action": cues[i]}
                                              for i in sorted(cues)]),
        solution_count=1,
    )
    return Entry(metadata, "\n".join(reference))


def _literals(yes, no):
    return ",".join([*yes, *(f"!{x}" for x in no)]) or "none"


@dataclass
class PlanningConfig(Config):
    horizon: int = 3

    def apply_difficulty(self, level):
        self.horizon += level


class Planning(Task):
    summary = "Find the unique cue-constrained plan that achieves a stated goal."
    task_name = "planning"
    task_version = 3
    config_cls = PlanningConfig

    def generate_entry(self):
        return _generate(self.config.level, self.config.horizon)

    def render_prompt(self, meta):
        rows = [f"{a.call}: {_literals(a.pre_true, a.pre_false)}"
                f" -> {_literals(a.add, a.delete)}" for a in meta.actions]
        steps = "; ".join(f"step {x.step} uses {x.action}" for x in meta.plan_cue.steps)
        cue = f"Cue: exactly {meta.horizon} actions" + (f"; {steps}." if steps else ".")
        return (
            f"Initial true facts: {', '.join(meta.initial_true)}. All other facts are false.\n\n"
            "Actions (preconditions -> effects; !fact means false):\n"
            f"{'\n'.join(rows)}\n\nGoal: {_literals(meta.goal_true, meta.goal_false)}.\n{cue}\n"
            "Return one grounded action per line."
        )

    def score_answer(self, answer, entry):
        meta = entry["metadata"]
        calls = [line.strip() for line in str(answer).splitlines() if line.strip()]
        cue = meta["plan_cue"]
        if len(calls) != cue["length"] or any(
                calls[x["step"] - 1] != x["action"] for x in cue["steps"]):
            return Reward(0, "plan cue mismatch")
        actions = {a["call"]: a for a in meta["actions"]}
        state = set(meta["initial_true"])
        for call in calls:
            action = actions.get(call)
            if action is None or not _applicable(action, state):
                return Reward(0, "invalid transition")
            state = _apply(action, state)
        valid = set(meta["goal_true"]) <= state and not set(meta["goal_false"]) & state
        return Reward(1) if valid else Reward(0, "goal not reached")
