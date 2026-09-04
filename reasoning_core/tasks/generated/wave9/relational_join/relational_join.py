import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'relational_join_execution (draw 1 of 1)',
 'hypothesis': 'HV-031',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/relational_join_execution',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2738823173,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class RelationalJoinConfig(Config):
    n_left: int = 4
    n_right: int = 4
    key_space: int = 3
    val_space: int = 8
    mode_pool: int = 0

    def apply_difficulty(self, level):
        self.n_left = sround(self.n_left + 2 * level)
        self.n_right = sround(self.n_right + 2 * level)
        self.key_space = sround(self.key_space + level)
        self.val_space = sround(self.val_space + 2 * level)
        self.mode_pool = min(4, self.mode_pool + level)


def _score_inner(la, ra, lb, rb, keys):
    total = 0
    for i in range(len(la)):
        for j in range(len(lb)):
            if la[i] == lb[j]:
                total += keys * ra[i] + rb[j]
    return total


def _score_left(la, ra, lb, rb, keys):
    total = 0
    for i in range(len(la)):
        matched = False
        for j in range(len(lb)):
            if la[i] == lb[j]:
                total += keys * ra[i] + rb[j]
                matched = True
        if not matched:
            total += keys * ra[i]
    return total


def _score_semi(la, ra, lb, rb, keys):
    total = 0
    for i in range(len(la)):
        if any(la[i] == lb[j] for j in range(len(lb))):
            total += keys * ra[i]
    return total


def _score_anti(la, ra, lb, rb, keys):
    total = 0
    for i in range(len(la)):
        if not any(la[i] == lb[j] for j in range(len(lb))):
            total += keys * ra[i]
    return total


MODES = [
    ("inner", _score_inner),
    ("left", _score_left),
    ("semi", _score_semi),
    ("anti", _score_anti),
]


class RelationalJoin(Task):
    summary = ("Execute inner, left, semi, and anti joins on small keyed relations "
               "with repeated keys, return a specified exact sum projection "
               "over an algebraic expression of key and value scalars.")
    config_cls = RelationalJoinConfig

    def generate_entry(self):
        cfg = self.config
        mode_count = cfg.mode_pool if cfg.mode_pool > 0 else 1
        mode_name, scorer = random.choice(MODES[:mode_count])

        la = [random.randrange(cfg.key_space) for _ in range(cfg.n_left)]
        ra = [random.randrange(1, cfg.val_space) for _ in range(cfg.n_left)]
        keys = random.randrange(1, 6)
        lb = [random.randrange(cfg.key_space) for _ in range(cfg.n_right)]
        rb = [random.randrange(1, cfg.val_space) for _ in range(cfg.n_right)]

        total = scorer(la, ra, lb, rb, keys)
        assert total >= 0

        cols_a = " ".join(str(v) for v in ra)
        cols_b = " ".join(str(v) for v in rb)
        keys_a = " ".join(str(v) for v in la)
        keys_b = " ".join(str(v) for v in lb)

        payload = {
            "A keys": keys_a,
            "A vals": cols_a,
            "B keys": keys_b,
            "B vals": cols_b,
            "join": mode_name,
            "how": "sum over matches of (K * A.val + B.val)"
                    if mode_name in ("inner", "left") else
                    ("sum of K * A.val over left rows with a match" if mode_name == "semi"
                     else "sum of K * A.val over left rows without a match"),
            "K": keys,
        }

        answer = str(total)
        metadata = edict({
            "left_keys": la,
            "left_vals": ra,
            "right_keys": lb,
            "right_vals": rb,
            "mode": mode_name,
            "K": keys,
            "total": total,
        })
        metadata.payload = payload
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (render_payload(metadata.payload) +
                "\n\nCompute the exact result of the described join. The answer "
                "is a single non-negative integer.")

    def score_answer(self, answer, entry):
        gt = entry.answer
        try:
            if int(answer) == int(gt):
                return 1.0
        except (ValueError, TypeError):
            return 0.0
        return 0.0
