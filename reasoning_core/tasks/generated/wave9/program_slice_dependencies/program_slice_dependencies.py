import random

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'program_slice_dependencies (draw 1 of 1)',
 'hypothesis': 'HV-029',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/program_slice_dependencies',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2807339826,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

_OPS = ("merge", "combine", "scale", "fold")


def _compute_slice(n, operands, guards, output):
    slice_set = set()
    work = [output]
    while work:
        s = work.pop()
        if s in slice_set:
            continue
        slice_set.add(s)
        for op in operands[s]:
            if op not in slice_set:
                work.append(op)
        g = guards[s]
        if g is not None and g not in slice_set:
            work.append(g)
    return slice_set


def _build_program(n, guard_prob):
    for _ in range(4000):
        operands = []
        guards = []
        for i in range(n):
            k = 0
            if i >= 1:
                k = random.randint(1, min(2, i))
            ops = sorted(random.sample(range(i), k)) if k else []
            used = set(ops)
            g = None
            remaining = [x for x in range(i) if x not in used]
            if remaining and random.random() < guard_prob:
                g = random.choice(remaining)
            operands.append(ops)
            guards.append(g)
        output = random.randrange(n)
        slice_set = _compute_slice(n, operands, guards, output)
        if 1 <= len(slice_set) <= n - 1:
            return operands, guards, output, slice_set
    raise RuntimeError("could not build a program with a proper slice")


class ProgramSliceDependenciesConfig(Config):
    n_stmts: int = 6
    guard_prob: float = 0.35

    def apply_difficulty(self, level):
        self.n_stmts = sround(self.n_stmts + 2 * level)
        self.guard_prob = min(0.55, 0.35 + 0.04 * level)


def _render_statement(i, ops, guard, n_ops, output):
    opname = _OPS[random.randrange(len(_OPS))]
    parts = []
    for x in ops:
        parts.append(f"v{x}")
    if random.random() < 0.4 and n_ops < 3:
        parts.append(str(random.randint(1, 9)))
    args = ", ".join(parts)
    target = f"v{i}"
    guard_text = ""
    if guard is not None:
        guard_text = f"if v{guard}: "
    return f"{i}: {guard_text}{target} = {opname}({args})"


class ProgramSliceDependencies(Task):
    summary = "Follow explicit data and control dependencies backward from a queried output, returning the canonical minimal relevant statement set."
    config_cls = ProgramSliceDependenciesConfig

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_stmts)
        guard_prob = float(cfg.guard_prob)
        operands, guards, output, slice_set = _build_program(n, guard_prob)
        lines = []
        for i in range(n):
            lines.append(_render_statement(i, operands[i], guards[i], len(operands[i]) + (1 if guards[i] is not None else 0), output))
        slice_list = sorted(slice_set)
        answer = " ".join(str(x) for x in slice_list)
        metadata = edict({
            'n_stmts': int(n),
            'output': int(output),
            'slice_list': [int(x) for x in slice_list],
        })
        metadata.payload = {
            'program': "\n".join(lines),
            'query': f"Final output value of v{output}",
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            "Trace the data and control dependencies backward from the final output "
            "and report the minimal set of statements that must be kept to compute "
            "that final value. Here 0-indexed statement numbers are shown at the "
            "start of each line; a line '3: if v2: v3 = merge(v1, 4)' means "
            "statement 3 depends on statement 1 (data) and on statement 2 (control, "
            "its guard). List the kept statement numbers in increasing order, "
            "separated by spaces (for example: '0 2 5'). The answer is exactly that list."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        text = str(answer).strip()
        if text == "":
            return 1.0 if entry.metadata.slice_list == [] else 0.0
        try:
            got = sorted(int(t) for t in text.replace(",", " ").split())
        except ValueError:
            return 0.0
        return 1.0 if got == entry.metadata.slice_list else 0.0
