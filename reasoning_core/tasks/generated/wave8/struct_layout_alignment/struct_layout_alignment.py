import ast
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'struct_layout_alignment (draw 2 of 2)',
 'hypothesis': 'W1-058',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/struct_layout_alignment',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 527083921,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

ABIS = ['x86', 'arm64', 'i386']


def _align_up(value, align):
    return (value + align - 1) // align * align


def _abi_align(abi, size):
    if abi == 'i386':
        return min(size, 4)
    if abi == 'arm64':
        return size if size < 16 else 16
    return size


def compute_layout(fields, abi):
    offsets = []
    offset = 0
    max_align = 1
    for size, align in fields:
        offset = _align_up(offset, align)
        offsets.append(offset)
        offset += size
        if align > max_align:
            max_align = align
    stride = _align_up(offset, max_align)
    return offsets, stride


@dataclass
class StructLayoutAlignmentConfig(Config):
    n_fields: int = 3
    max_size: int = 8
    strided_prob: float = 0.5

    def apply_difficulty(self, level):
        self.n_fields = 3 + level
        self.max_size = 4 + level
        self.strided_prob = 0.5


class StructLayoutAlignment(Task):
    summary = ("Given field sizes, alignments, and ABI rules (x86 equal, "
               "arm64 cap 16, i386 cap 4), output field offsets or the total "
               "struct stride; align each field to its alignment and pad the "
               "struct to its maximum field alignment.")
    config_cls = StructLayoutAlignmentConfig

    def generate_entry(self):
        cfg = self.config
        abi = random.choice(ABIS)
        strided = random.random() < cfg.strided_prob
        n = max(int(cfg.n_fields), 1)
        fields = []
        for _ in range(n):
            size = random.randint(1, int(cfg.max_size))
            align = _abi_align(abi, size)
            fields.append((size, align))
        offsets, stride = compute_layout(fields, abi)
        self._check(fields, abi, offsets, stride)
        if strided:
            answer = str(stride)
        else:
            answer = str(offsets)
        metadata = edict({
            'fields': [[int(s), int(a)] for s, a in fields],
            'abi': abi,
            'strided': bool(strided),
            'offsets': [int(o) for o in offsets],
            'stride': int(stride),
        })
        metadata.payload = {
            'fields': metadata['fields'],
            'abi': abi,
            'strided': metadata['strided'],
        }
        entry = Entry(metadata=metadata, answer=answer)
        return entry

    def _check(self, fields, abi, offsets, stride):
        recompute, rstride = compute_layout(fields, abi)
        assert recompute == offsets
        assert rstride == stride
        for i, (o, (s, a)) in enumerate(zip(offsets, fields)):
            assert o % a == 0
        assert stride % max([1] + [a for _, a in fields]) == 0
        assert all(o >= 0 for o in offsets)

    def render_prompt(self, metadata):
        fields = ', '.join(
            f"size {s} (alignment {a})" for s, a in metadata.fields
        )
        if metadata.strided:
            target = "the total size (stride) of the struct, as a single integer"
        else:
            target = "the offset of each field, as a list of integers"
        return (
            f"A struct is laid out under the {metadata.abi} ABI with fields: "
            f"{fields}. Align each field's start offset up to its alignment; "
            "after the last field, pad the struct up to the largest field "
            f"alignment. Give {target}. Answer with no extra text."
        )

    def score_answer(self, answer, entry):
        truth = entry.answer
        if entry.metadata['strided']:
            try:
                val = int(str(answer).strip())
            except Exception:
                return 0.0
            return 1.0 if val == int(truth) else 0.0
        try:
            val = ast.literal_eval(str(answer).strip())
        except Exception:
            return 0.0
        if not isinstance(val, list):
            return 0.0
        truth_list = ast.literal_eval(truth)
        if [int(x) for x in val] == truth_list:
            return 1.0
        return 0.0
