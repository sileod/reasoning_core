from dataclasses import dataclass
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload


@dataclass
class ClosureCaptureConfig(Config):
    n_cells: int = 2
    n_records: int = 3

    def apply_difficulty(self, level):
        self.n_cells = 2 + level
        self.n_records = 3 + level


def _resnames(n):
    names = []
    while len(names) < n:
        c = chr(ord("a") + random.randrange(26))
        if c not in names:
            names.append(c)
    return names


class ClosureCaptureResolution(Task):
    summary = ("Track lexical captures through closure creation, later rebinding, "
               "and invocation, returning the value observed through the captured environment; "
               "mix direct, rebound-before-call, and loop-created closures over multiple cells.")

    config_cls = ClosureCaptureConfig

    def generate_entry(self):
        while True:
            entry = self._build()
            if entry is not None:
                return entry

    def _build(self):
        n_cells = int(self.config.n_cells)
        n_records = int(self.config.n_records)

        names = _resnames(n_cells)
        env = {name: random.randrange(-9, 10) for name in names}

        lines = [f"{name} = {env[name]}" for name in names]
        snapshots = {}
        call = None
        answer = None

        def new_closure(name):
            snapshots[name] = env[name]

        for _ in range(n_records):
            which = random.randrange(3)
            name = random.choice(names)
            if which == 0:
                lines.append(f"f = (lambda: {name})")
                new_closure(name)
            elif which == 1:
                lines.append(f"g = (lambda: {name})")
                lines.append(f"{name} = {random.randrange(-9, 10)}")
                new_closure(name)
            else:
                loops = random.randrange(2, 5)
                for _ in range(loops):
                    lines.append(f"h = (lambda: {name})")
                    new_closure(name)

        which = random.randrange(3)
        target_name = random.choice(names)
        if which == 0:
            lines.append(f"f = (lambda: {target_name})")
            new_closure(target_name)
            call = "f()"
            last_kind = "direct"
        elif which == 1:
            lines.append(f"g = (lambda: {target_name})")
            new_closure(target_name)
            env[target_name] = random.randrange(-9, 10)
            lines.append(f"{target_name} = {env[target_name]}")
            call = "g()"
            last_kind = "rebound_after"
        else:
            loops = random.randrange(2, 5)
            for _ in range(loops):
                lines.append(f"h = (lambda: {target_name})")
                new_closure(target_name)
            call = "h()"
            last_kind = "loop"

        answer = snapshots[target_name]
        if answer < -100 or answer > 100:
            return None

        simulated = _simulate(lines, call)
        if simulated != answer:
            return None

        nums = []
        for ln in lines:
            ln = ln.strip()
            if " = " in ln and "(lambda" not in ln:
                nums.append(int(ln.split(" = ")[1]))
        if nums:
            if answer in (nums[0], nums[-1], max(nums), min(nums)):
                return None

        metadata = edict({
            "n_cells": n_cells,
            "n_records": n_records,
            "invocation_kind": last_kind,
            "call": call,
            "target": target_name,
        })
        metadata.payload = {"program": "\n".join(lines), "call": call}
        metadata.program_lines = lines
        metadata.env_at_call = dict(env)

        return Entry(metadata=metadata, answer=str(answer))

    def render_prompt(self, metadata):
        return (
            "Each lambda (lambda: X) captures the CURRENT value of X at the moment "
            "the lambda is created; later assignments to X do not change what an "
            "already-created lambda sees.\n\n"
            + render_payload(metadata.payload)
            + "\n\nWhat is the value of " + metadata.call + "?"
        )

    def score_answer(self, answer, entry):
        try:
            return 1.0 if int(str(answer).strip()) == int(entry.answer) else 0.0
        except (ValueError, TypeError):
            return 0.0


def _simulate(lines, call):
    env = {}
    closures = {}
    for ln in lines:
        ln = ln.strip()
        if " = (lambda: " in ln:
            lhs, _, rhs = ln.partition(" = (lambda: ")
            var = rhs.rstrip(")")
            closures[lhs] = env[var]
        elif " = " in ln:
            lhs, _, rhs = ln.partition(" = ")
            env[lhs] = int(rhs)
    cname = call.split("(")[0]
    return closures[cname]


TASK_META = {'parent_source_id': None,
 'idea': 'closure_capture_resolution (draw 1 of 1)',
 'hypothesis': 'HV-026',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/closure_capture_resolution',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1322404106,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
