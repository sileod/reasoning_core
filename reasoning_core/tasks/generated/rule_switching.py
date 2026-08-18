import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


PRIMITIVES = {
    "rotate-left": (1, 2, 0),
    "rotate-right": (2, 0, 1),
    "swap-first-two": (1, 0, 2),
    "swap-last-two": (0, 2, 1),
    "swap-outer": (2, 1, 0),
}
OPCODES = ("X", "Y", "Z")


def _apply(state, regs, perm):
    old = [state[r] for r in regs]
    for dst, src in enumerate(perm):
        state[regs[dst]] = old[src]


def _lineage(target, operations):
    reg = target
    relevant = []
    for index in range(len(operations) - 1, -1, -1):
        op = operations[index]
        regs = op["regs"]
        if reg not in regs:
            continue
        dst = regs.index(reg)
        previous = regs[op["perm"][dst]]
        if previous != reg:
            relevant.append(index)
        reg = previous
    relevant.reverse()
    return relevant


def _has_switch_interference(operations, relevant):
    by_opcode = {}
    for i in relevant:
        op = operations[i]
        by_opcode.setdefault(op["opcode"], []).append((op["mode"], op["primitive"]))
    for uses in by_opcode.values():
        if len({mode for mode, _ in uses}) >= 2 and len({primitive for _, primitive in uses}) >= 2:
            return True
    return False


@dataclass
class RuleSwitchingConfig(Config):
    n_registers: int = 5
    n_steps: int = 7
    n_modes: int = 2
    min_dependency_depth: int = 2

    def apply_difficulty(self, level):
        self.n_registers = sround(self.n_registers + 0.5 * level)
        self.n_steps = sround(self.n_steps + 1.8 * level)
        self.n_modes = sround(self.n_modes + 0.3 * level)
        self.min_dependency_depth = sround(self.min_dependency_depth + 0.6 * level)


class RuleSwitching(GeneratedMixin, Task):
    summary = "Track symbolic state while identical opcodes change meaning across active rule modes."
    config_cls = RuleSwitchingConfig

    def generate_entry(self):
        n_registers = min(9, max(4, int(self.config.n_registers)))
        n_steps = max(5, int(self.config.n_steps))
        n_modes = min(4, max(2, int(self.config.n_modes)))
        min_depth = min(n_steps - 1, max(2, int(self.config.min_dependency_depth)))
        registers = [f"r{i + 1}" for i in range(n_registers)]
        values = [chr(ord("A") + i) for i in range(n_registers)]

        for _ in range(400):
            mappings = []
            for mode in range(n_modes):
                for _ in range(50):
                    chosen = random.sample(list(PRIMITIVES), len(OPCODES))
                    mapping = dict(zip(OPCODES, chosen))
                    if mode == 0 or mapping["X"] != mappings[-1]["X"]:
                        mappings.append(mapping)
                        break

            state = dict(zip(registers, values))
            current_mode = 0
            operations = []
            program = []
            switch_every = max(1, n_steps // (n_modes + 1))

            for step in range(n_steps):
                if step and (step % switch_every == 0 or random.random() < 0.18):
                    choices = [m for m in range(n_modes) if m != current_mode]
                    current_mode = random.choice(choices)
                    program.append({"kind": "mode", "mode": current_mode})
                opcode = random.choices(OPCODES, weights=[3, 1, 1])[0]
                regs = tuple(random.sample(registers, 3))
                primitive = mappings[current_mode][opcode]
                perm = PRIMITIVES[primitive]
                operations.append({
                    "mode": current_mode,
                    "opcode": opcode,
                    "regs": regs,
                    "primitive": primitive,
                    "perm": perm,
                })
                program.append({"kind": "op", "operation": len(operations) - 1})
                _apply(state, regs, perm)

            candidates = []
            for target in registers:
                relevant = _lineage(target, operations)
                if len(relevant) >= min_depth and _has_switch_interference(operations, relevant):
                    candidates.append((len(relevant), target, relevant))
            if not candidates:
                continue
            depth, target, relevant = random.choice([x for x in candidates if x[0] == max(y[0] for y in candidates)])
            return Entry(
                metadata=edict(
                    registers=registers,
                    initial=dict(zip(registers, values)),
                    mappings=mappings,
                    program=program,
                    operations=operations,
                    target=target,
                    dependency_depth=depth,
                    relevant_operations=relevant,
                ),
                answer=state[target],
            )
        raise RuntimeError("RuleSwitching: could not build a sufficiently interfering trace")

    def render_prompt(self, m):
        mapping_lines = []
        for i, mapping in enumerate(m.mappings, 1):
            mapping_lines.append(f"M{i}: " + ", ".join(f"{op}={mapping[op]}" for op in OPCODES))
        program_lines = []
        for item in m.program:
            if item["kind"] == "mode":
                program_lines.append(f"mode M{item['mode'] + 1}")
            else:
                op = m.operations[item["operation"]]
                program_lines.append(f"{op['opcode']} " + " ".join(op["regs"]))
        initial = " ".join(f"{r}={m.initial[r]}" for r in m.registers)
        return (
            "Maintain the register values while executing the program. The current mode determines each opcode's "
            "meaning. For an instruction on registers (a,b,c): rotate-left maps their values to (b,c,a); "
            "rotate-right to (c,a,b); swap-first-two to (b,a,c); swap-last-two to (a,c,b); and swap-outer to (c,b,a). "
            "Mode changes affect following instructions.\n"
            f"Modes:\n" + "\n".join(mapping_lines) + "\n"
            f"Initial state: {initial}\nStart mode: M1\nProgram:\n" + "\n".join(program_lines) + "\n"
            f"What value is in {m.target} after the program? The answer is one value label."
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip().upper() == str(entry.answer).strip().upper())

    def balancing_key(self, problem):
        return problem.answer
