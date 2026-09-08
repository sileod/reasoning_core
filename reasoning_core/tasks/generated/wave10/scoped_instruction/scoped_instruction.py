import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict


@dataclass
class ScopedInstructionConfig(Config):
    level: int = 0
    max_value: int = 20
    section_len: int = 3

    def apply_difficulty(self, level):
        self.level = level
        self.max_value = 20 + 10 * level
        self.section_len = 3 + level


class ScopedInstruction(Task):
    summary = "Apply a single-item, single-section, or single-turn instruction against original defaults and report the value or sequence the scoped edit yields."

    config_cls = ScopedInstructionConfig

    MODES = ["item", "section", "turn"]

    def _seq(self, config):
        n = config.section_len
        if config.level >= 4:
            n = config.section_len + 2
        max_value = config.max_value
        return [int(random.randrange(1, max_value + 1)) for _ in range(n)]

    def generate_entry(self):
        config = self.config
        mode = self.MODES[int(random.randrange(len(self.MODES)))]
        max_value = config.max_value
        payload = edict({"mode": mode})

        base = self._seq(config)
        S = sum(base)
        payload.base = list(base)

        if mode == "section":
            new_vals = [int(random.randrange(S + 1, S + 1 + max_value)) for _ in base]
            payload.new = list(new_vals)
            answer = ",".join(str(v) for v in new_vals)
            cot = (f"Instruction covers the whole section, so every default is overridden by the given new values. "
                   f"Resulting sequence is {new_vals}.")
        else:
            idx = int(random.randrange(len(base)))
            payload.idx = idx
            target = S + int(random.randrange(1, max_value + 1))
            payload.target = target
            working = list(base)
            working[idx] = target
            answer = str(target)
            if mode == "item":
                cot = (f"Instruction is limited to item {idx}: that item becomes {target}; "
                       f"every other item is restored to its original default. The changed item is {target}. "
                       f"Sequence: {working}.")
            else:
                cot = (f"The turn edits only slot {idx} to {target}, restoring the other slots to their "
                       f"original defaults. The changed slot holds {target}. Sequence: {working}.")

        meta = edict({"cot": cot})
        meta.payload = payload
        return Entry(metadata=meta, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        base = ", ".join(str(v) for v in p.base)
        if p.mode == "section":
            new_block = ", ".join(str(v) for v in p.new)
            return (f"The sequence records a default value in each slot, in order. "
                    f"The current defaults are {base}. "
                    f"Apply an instruction that covers the entire section at once: every slot is "
                    f"overwritten with a fresh value, and the fresh values, in slot order, are {new_block}. "
                    f"No default survives the edit. What is the resulting sequence, as the comma-separated "
                    f"list in slot order?")
        idx = p.idx
        if p.mode == "item":
            return (f"The sequence records a default value in each slot, in order. "
                    f"The current defaults are {base}. "
                    f"Apply an instruction that is scoped to a single item, slot {idx}: replace the value in "
                    f"that slot with the sum of every default in the sequence plus an additional increment of "
                    f"{p.target - sum(p.base)}, while every other slot is restored to its original default. "
                    f"What numeric value ends up in slot {idx}?")
        return (f"The sequence records a default value in each slot, in order. "
                f"The current defaults are {base}. "
                f"A single-turn instruction is limited to slot {idx}: set that slot to the sum of every "
                f"default plus an increment of {p.target - sum(p.base)}, and restore every other slot to its "
                f"original default. What numeric value ends up in slot {idx}?")

    def score_answer(self, answer, entry):
        try:
            a = answer.strip()
        except Exception:
            return 0.0
        p = entry.metadata.payload
        if p.mode == "section":
            parts = [x.strip() for x in a.split(",")]
            expected = [str(v) for v in p.new]
            if parts == expected:
                return 1.0
            return 0.0
        try:
            val = int(a)
        except Exception:
            return 0.0
        return 1.0 if val == p.target else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'scoped_instruction (draw 1 of 2)',
 'hypothesis': 'ASTRA0-06',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/scoped_instruction',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4293640035,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
