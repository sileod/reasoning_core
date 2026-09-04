import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround
from reasoning_core.utils import score_scalar

TASK_META = {'parent_source_id': None,
 'idea': 'Add binary-heap simulation over a sequence of operations.',
 'hypothesis': 'S10',
 'changes': 'Apply pushes, pops and decrease-keys and ask for the resulting '
            'array.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 747576363,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class HeapSimulationConfig(Config):
    n_init: int = 4
    n_ops: int = 3
    max_val: int = 30
    ask_index: bool = False

    def apply_difficulty(self, level):
        self.n_init = sround(self.n_init + level)
        self.n_ops = sround(self.n_ops + 2 * level)
        self.max_val = sround(self.max_val + 20 * level)
        self.ask_index = level >= 3


def _sift_up(a, i):
    while i > 0:
        p = (i - 1) // 2
        if a[i] < a[p]:
            a[i], a[p] = a[p], a[i]
            i = p
        else:
            break


def _sift_down(a, n, i):
    while True:
        l = 2 * i + 1
        if l >= n:
            break
        r = l + 1
        m = l
        if r < n and a[r] < a[l]:
            m = r
        if a[m] < a[i]:
            a[i], a[m] = a[m], a[i]
            i = m
        else:
            break


class HeapSimulation(Task):
    config_cls = HeapSimulationConfig

    def _heapify(self, a):
        n = len(a)
        for i in range(n // 2 - 1, -1, -1):
            _sift_down(a, n, i)

    def _push(self, a, v):
        a.append(v)
        _sift_up(a, len(a) - 1)

    def _pop_min(self, a):
        a[0] = a[-1]
        a.pop()
        if a:
            _sift_down(a, len(a), 0)

    def _replace_root(self, a, v):
        a[0] = v
        _sift_down(a, len(a), 0)

    def generate_entry(self):
        cfg = self.config
        while True:
            init = [random.randint(0, cfg.max_val) for _ in range(cfg.n_init)]
            a = list(init)
            self._heapify(a)
            ops = []
            final = list(a)
            for _ in range(cfg.n_ops):
                kind = random.choice(['push', 'pop', 'replace'])
                if kind == 'push':
                    v = random.randint(0, cfg.max_val)
                    if not final:
                        final.append(v)
                    else:
                        self._push(final, v)
                    ops.append(('push', v))
                elif kind == 'pop':
                    if final:
                        self._pop_min(final)
                    ops.append(('pop',))
                else:
                    v = random.randint(0, cfg.max_val)
                    if not final:
                        final.append(v)
                    else:
                        self._replace_root(final, v)
                    ops.append(('replace', v))
            if final:
                break
        if cfg.ask_index:
            index = random.randrange(len(final))
            answer = str(final[index])
        else:
            answer = ','.join(str(x) for x in final)
        metadata = edict({
            'init': init,
            'heap': list(a),
            'ops': ops,
            'ask_index': cfg.ask_index,
            '_index': index if cfg.ask_index else None,
        })
        metadata.payload = {
            'init': init,
            'ops': ops,
            'ask_index': cfg.ask_index,
            'index': index if cfg.ask_index else None,
        }
        metadata['_answer'] = answer
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        ops_str = '; '.join(self._render_op(op) for op in metadata.ops)
        if metadata.ask_index:
            return (f"We maintain a min-heap as an array. Operations use sift-down for the "
                    f"smallest-child convention (the root is the minimum, children at "
                    f"2i+1 and 2i+2), and sift-up for the parent at (i-1)//2. Start from the "
                    f"array {metadata.init} (heapified). Then apply, in order: {ops_str}. "
                    f"Give the value at index {metadata['_index']} of the resulting array.\n"
                    f"The answer is a single integer.")
        return (f"We maintain a min-heap as an array. Operations use sift-down for the "
                f"smallest-child convention (the root is the minimum, children at 2i+1 and "
                f"2i+2), and sift-up for the parent at (i-1)//2. Start from the array "
                f"{metadata.init} (heapified). Then apply, in order: {ops_str}. Give the "
                f"resulting array.\nThe answer is a comma-separated list of integers.")

    def _render_op(self, op):
        if op[0] == 'push':
            return f"push {op[1]}"
        if op[0] == 'pop':
            return "pop-min"
        return f"replace-root {op[1]}"

    def score_answer(self, answer, entry):
        gt = entry.answer
        if entry.metadata.ask_index:
            return score_scalar(answer, entry)
        answer = answer.strip()
        try:
            got = [int(x) for x in answer.split(',') if x.strip() != '']
        except ValueError:
            return 0.0
        expected = [int(x) for x in gt.split(',')]
        if got == expected:
            return 1.0
        return 0.0
