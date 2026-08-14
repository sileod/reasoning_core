import ast
import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround
from ._base import GeneratedMixin, exact


@dataclass
class ControlledCodeExecutionConfig(Config):
    n_segments: int = 2
    magnitude: int = 4
    list_size: int = 3

    def apply_difficulty(self, level):
        self.n_segments = sround(self.n_segments + 0.6 * level)
        self.magnitude = sround(self.magnitude + 0.8 * level)
        self.list_size = sround(self.list_size + 0.3 * level)


def _run_endpoint(code):
    ns = {"__builtins__": {"range": range, "len": len, "sum": sum, "min": min, "max": max}}
    exec(compile(code, "<controlled-python>", "exec"), ns, ns)
    return repr(ns["endpoint"]())


def _controlled_program(cfg):
    n = max(2, cfg.list_size)
    mag = max(2, cfg.magnitude)
    initial = [random.randint(-mag, mag) for _ in range(n)]
    segments = random.sample(
        ["alias", "closure", "default", "mutation", "loop", "rebind"],
        k=min(max(2, cfg.n_segments), 6),
    )
    lines = ["def endpoint():", f"    state = {initial!r}"]
    phenomena = []

    for k, kind in enumerate(segments):
        i, j = random.sample(range(n), 2)
        a = random.choice([x for x in range(-mag, mag + 1) if x])
        b = random.choice([x for x in range(-mag, mag + 1) if x])

        if kind == "alias":
            lines += [
                f"    alias{k} = state",
                f"    alias{k}[{i}] += {a}",
                f"    state[{j}] += alias{k}[{i}]",
            ]
            phenomena.append("aliasing")
        elif kind == "closure":
            lines += [
                f"    bias{k} = {a}",
                f"    def f{k}(x):",
                f"        return x + bias{k} + state[{j}]",
                f"    bias{k} += {b}",
                f"    state[{i}] = f{k}(state[{i}])",
            ]
            phenomena.append("closure late binding")
        elif kind == "default":
            lines += [
                f"    bias{k} = {a}",
                f"    def f{k}(x, bias=bias{k}):",
                f"        return x + bias + state[{j}]",
                f"    bias{k} += {b}",
                f"    state[{i}] = f{k}(state[{i}])",
            ]
            phenomena.append("default-argument capture")
        elif kind == "mutation":
            lines += [
                f"    def f{k}(xs, d):",
                f"        xs[{i}] += d",
                f"        return xs[{i}] - xs[{j}]",
                f"    state[{j}] += f{k}(state, {a})",
            ]
            phenomena.append("mutation through calls")
        elif kind == "loop":
            vals = [random.randint(-mag, mag) for _ in range(random.randint(2, 4))]
            mul = random.choice([-2, -1, 1, 2])
            lines += [
                f"    for v{k} in {vals!r}:",
                f"        state[{i}] = {mul} * state[{i}] + v{k}",
                f"        state[{j}] += state[{i}]",
            ]
            phenomena.append("loop-carried state")
        else:
            lines += [
                f"    alias{k} = state",
                f"    state = state[:]",
                f"    state[{i}] += {a}",
                f"    alias{k}[{j}] += state[{i}]",
                f"    state[{j}] += alias{k}[{i}]",
            ]
            phenomena.append("rebinding versus aliasing")

    lines.append("    return state")
    return "\n".join(lines) + "\n", phenomena


class ControlledCodeExecution(GeneratedMixin, Task):
    summary = "Execute Python programs generated to require controlled semantic phenomena."
    config_cls = ControlledCodeExecutionConfig

    def generate_entry(self):
        code, phenomena = _controlled_program(self.config)
        answer = _run_endpoint(code)
        metadata = edict(code=code, phenomena=phenomena)
        metadata.payload = {"code": code}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            "Predict the value returned by this Python call.\n"
            f"```python\n{metadata.code}```\n"
            "Call: `endpoint()`\n"
            "The answer is the exact Python `repr` of the returned value."
        )

    def score_answer(self, answer, entry):
        try:
            a = ast.literal_eval(str(answer).strip())
            b = ast.literal_eval(str(entry.answer).strip())
            return float(a == b)
        except Exception:
            return exact(answer, entry)

    def balancing_key(self, problem):
        return tuple(problem.metadata.phenomena[:2])
