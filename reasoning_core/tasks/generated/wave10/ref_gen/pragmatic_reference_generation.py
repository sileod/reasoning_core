import random
from itertools import combinations

from reasoning_core.template import Task, Entry, Config, edict, render_payload


TASK_META = {'parent_source_id': None,
 'idea': 'pragmatic_reference_generation (draw 1 of 2)',
 'hypothesis': 'ASTRA0-20',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/pragmatic_reference_generation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3457176975,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


SHAPES = ["circle", "square", "triangle", "star", "hexagon"]
COLORS = ["red", "green", "blue", "yellow", "purple"]
PATTERNS = ["solid", "striped", "dotted", "checkered"]

ATTR_ORDER = ["shape", "color", "pattern"]


def _doc(obj):
    return "; ".join(f"{a}={obj[a]}" for a in ATTR_ORDER)


def _render_objects(objs):
    return "\n".join(f"{o['name']}: {_doc(o)}" for o in objs)


def _identifies(attrs, target, objs):
    return sum(
        1 for o in objs if all(o[a] == target[a] for a in attrs)
    ) == 1


class RefConfig(Config):
    n_obj: int = 4

    def apply_difficulty(self, level):
        self.n_obj = 4 + level


class RefGenTask(Task):
    task_name = "ref_gen"
    summary = ("Produce the shortest description uniquely identifying an object among "
               "distractors in a shared scene.")
    config_cls = RefConfig

    def generate_entry(self):
        n = self.config.n_obj
        objs = []
        used = set()
        attempts = 0
        while len(objs) < n and attempts < 5000:
            attempts += 1
            obj = {
                "name": f"object{len(objs)+1}",
                "shape": random.choice(SHAPES),
                "color": random.choice(COLORS),
                "pattern": random.choice(PATTERNS),
            }
            key = (obj["shape"], obj["color"], obj["pattern"])
            if key in used:
                continue
            used.add(key)
            objs.append(obj)

        objs.sort(key=lambda o: o["name"])
        target = random.choice(objs)

        single = []
        for a in ATTR_ORDER:
            if sum(1 for o in objs if o[a] == target[a]) == 1:
                single.append(a)

        if single:
            chosen = [random.choice(single)]
        else:
            chosen = None
            for r in range(1, len(ATTR_ORDER) + 1):
                hit = None
                for combo in combinations(ATTR_ORDER, r):
                    if _identifies(combo, target, objs):
                        hit = combo
                        break
                if hit is not None:
                    chosen = list(hit)
                    break
            if chosen is None:
                raise RuntimeError("no identifying description found")

        answer = "; ".join(f"{a}={target[a]}" for a in chosen)
        assert _identifies(chosen, target, objs), f"answer {answer} not identifying"

        metadata = edict({
            "objects": objs,
            "chosen_attrs": chosen,
            "target_name": target["name"],
        })
        metadata.payload = {"scene": _render_objects(objs)}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return (payload + "\n\nWrite the shortest description that refers to exactly one of "
                "the objects above. Give the answer in the same 'attr=value; attr=value' "
                "format, naming the fewest attributes possible that pick out that one object.")

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        a = _normalize(answer)
        t = _normalize(entry.answer)
        return 1.0 if a == t else 0.0


def _normalize(s):
    return " ".join(s.lower().replace(";", " ").split())
