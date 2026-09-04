import ast
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class AliasMutationConfig(Config):
    n_vars: int = 4
    n_ops: int = 6
    max_val: int = 9
    max_len: int = 4

    def apply_difficulty(self, level):
        self.n_vars = sround(self.n_vars + level)
        self.n_ops = sround(self.n_ops + 2 * level)
        self.max_val = sround(self.max_val + 2 * level)
        self.max_len = sround(self.max_len + level)


def _parse_list(text):
    try:
        value = ast.literal_eval(text.strip())
    except Exception:
        return None
    if isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        return value
    return None


class AliasMutation(Task):
    summary = ("Track aliases of mutable lists across assignment, in-place "
               "append/index mutation, and rebinding, returning a queried final list value.")
    config_cls = AliasMutationConfig

    def generate_entry(self):
        cfg = self.config
        n = max(3, int(cfg.n_vars))
        names = [chr(ord('a') + i) for i in range(n)]

        vars_obj = {}
        init = {}
        for nm in names:
            L = [random.randint(0, int(cfg.max_val))
                 for _ in range(random.randint(1, max(1, int(cfg.max_len))))]
            vars_obj[nm] = L
            init[nm] = list(L)

        lines = []
        touched = set()
        n_ops = max(3, int(cfg.n_ops))
        for _ in range(n_ops):
            kind = random.choice(['rebind', 'alias', 'append', 'setitem'])
            v = random.choice(names)
            if kind == 'alias':
                w = random.choice([x for x in names if x != v])
                vars_obj[v] = vars_obj[w]
                lines.append(f"{v} = {w}")
                touched.add(v)
                touched.add(w)
            elif kind == 'append':
                x = random.randint(0, int(cfg.max_val))
                vars_obj[v].append(x)
                lines.append(f"{v}.append({x})")
                touched.add(v)
            elif kind == 'setitem':
                if len(vars_obj[v]) == 0:
                    continue
                i = random.randrange(len(vars_obj[v]))
                x = random.randint(0, int(cfg.max_val))
                vars_obj[v][i] = x
                lines.append(f"{v}[{i}] = {x}")
                touched.add(v)
            else:  # rebind
                elems = [random.randint(0, int(cfg.max_val))
                         for _ in range(random.randint(1, max(1, int(cfg.max_len))))]
                vars_obj[v] = list(elems)
                lines.append(f"{v} = {elems}")
                touched.add(v)

        if touched:
            query = sorted(touched)[0]
        else:
            query = names[0]

        gold = list(vars_obj[query])

        metadata = edict({
            'variables': init,
            'operations': lines,
            'query': query,
            'answer': gold,
        })
        metadata.payload = {
            'variables': init,
            'operations': lines,
            'query': query,
        }
        return Entry(metadata=metadata, answer=str(gold))

    def render_prompt(self, metadata):
        lines = [f"{k} = {v}" for k, v in metadata.variables.items()]
        ops = "\n".join(f"{i + 1}. {op}" for i, op in enumerate(metadata.operations))
        return (f"Variables begin as:\n"
                + "\n".join(lines)
                + "\n\nPerform these operations in order:\n"
                + ops
                + f"\n\nAfter all operations, what is the final value of variable {metadata.query}?\n\n"
                + "The answer is a list literal, e.g. [1, 2, 3].")

    def score_answer(self, answer, entry):
        gold = _parse_list(entry.answer)
        got = _parse_list(answer)
        if gold is None or got is None:
            return 0.0
        return 1.0 if got == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'alias_mutation_tracking (draw 1 of 1)',
 'hypothesis': 'HV-023',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/alias_mutation_tracking',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1337854016,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
