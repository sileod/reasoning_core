import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

LEVELS = ["employee", "manager", "director", "ceo", "regulation"]
_ROLE_LABEL = {
    "employee": "an employee",
    "manager": "a manager",
    "director": "a director",
    "ceo": "the CEO",
    "regulation": "regulation",
}
_PARAMS = ["speed", "power", "angle", "delay"]

TASK_META = {'parent_source_id': None,
 'idea': 'instruction_priority (draw 1 of 2)',
 'hypothesis': 'ASTRA0-01',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/instruction_priority',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3427748574,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class PriorityConfig(Config):
    n_instructions: int = 4

    def apply_difficulty(self, level):
        self.n_instructions = 3 + level


def _resolve(param_instructions):
    best_level = None
    best_idx = -1
    for idx, (level, value) in enumerate(param_instructions):
        if best_level is None or _RANK[level] > _RANK[best_level] or (
                _RANK[level] == _RANK[best_level] and idx > best_idx):
            best_level = level
            best_idx = idx
    return param_instructions[best_idx][1]


_RANK = {role: i for i, role in enumerate(LEVELS)}


class Priority(Task):
    summary = "Answer under conflicting instructions from different authority levels; lower-priority instructions still apply where compatible."
    config_cls = PriorityConfig

    def generate_entry(self):
        metric = {
            "last_number": lambda dec: dec[0] == len(orders) - 1,
            "first_number": lambda dec: dec[0] == 0,
        }
        for _ in range(40):
            n = self.config.n_instructions
            n_params = random.randint(2, min(3, n))
            target = random.choice(_PARAMS[:n_params])

            orders = []
            for _ in range(n):
                p = random.choice(_PARAMS[:n_params])
                level = random.choice(LEVELS)
                value = random.randint(1, 20)
                orders.append((p, level, value))

            while len({p for p, _, _ in orders}) < n_params:
                missing = [p for p in _PARAMS[:n_params]
                           if p not in {pp for pp, _, _ in orders}]
                orders[random.randrange(n)] = (
                    random.choice(missing), random.choice(LEVELS),
                    random.randint(1, 20))

            random.shuffle(orders)

            def decide():
                d = None
                for idx, (pp, lvl, v) in enumerate(orders):
                    if pp == target:
                        if d is None or _RANK[lvl] > _RANK[d[1][1]] or (
                                _RANK[lvl] == _RANK[d[1][1]] and idx > d[0]):
                            d = (idx, (pp, lvl, v))
                return d

            dec = decide()
            if len(orders) >= 3:
                anchor = random.randrange(1, len(orders) - 1)
                di = dec[0]
                if di != anchor:
                    orders[di], orders[anchor] = orders[anchor], orders[di]
            dec = decide()

            values = [v for _, _, v in orders]
            ans_val = dec[1][2]
            if ans_val in (max(values), min(values)) and len(set(values)) > 1:
                continue
            if metric["last_number"](dec) or metric["first_number"](dec):
                continue

            final_values = {}
            for p in _PARAMS[:n_params]:
                final_values[p] = _resolve(
                    [(lvl, v) for pp, lvl, v in orders if pp == p])

            sentences = []
            for p, level, value in orders:
                sentences.append(
                    f"{_ROLE_LABEL[level]} says to set {p} to {value}")

            answer = str(final_values[target])

            payload = {
                "situation": (
                    "A team must configure a machine. Instructions arrive in the "
                    "order listed. A manager's instruction overrides an employee's, "
                    "a director's overrides a manager's, the CEO's overrides a "
                    "director's, and regulation overrides everyone. When two "
                    "instructions of the same authority touch the same setting, the "
                    "later one in the list governs. Lower-authority instructions "
                    "still apply to any setting no higher instruction touches."
                ),
                "orders": "; ".join(sentences) + ".",
                "task": f"What is the final value of {target}?",
            }
            metadata = edict(payload)
            metadata.payload = dict(payload)
            metadata.answer = answer
            metadata.final = {k: str(v) for k, v in final_values.items()}
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("could not build a well-separated instance")

    def render_prompt(self, metadata):
        return (
            f"{metadata.payload['situation']}\n"
            f"{metadata.payload['orders']}\n"
            f"{metadata.payload['task']}\n"
            "The answer is an integer."
        )

    def score_answer(self, answer, entry):
        a = str(answer).strip()
        gold = str(entry.answer).strip()
        if a == gold:
            return 1.0
        try:
            return 1.0 if int(a) == int(gold) else 0.0
        except (ValueError, TypeError):
            return 0.0
