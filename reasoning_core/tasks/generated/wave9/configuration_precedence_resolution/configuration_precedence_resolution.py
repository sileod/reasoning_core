"""Configuration precedence resolution task.

Merge defaults, inherited settings, environment overrides, local overrides, and
deletion markers to return one effective configuration.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'configuration_precedence_resolution (draw 1 of 1)',
 'hypothesis': 'HV-073',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/configuration_precedence_resolution',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1903761430,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

KEY_CHARS = "abcdefghijklmnopqrstuvwxyz"
VAL_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LAYER_NAMES = ("defaults", "inherited", "environment", "local")


def _rand_key(r):
    return ''.join(r.choice(KEY_CHARS) for _ in range(r.randint(2, 5)))


def _rand_val(r):
    return ''.join(r.choice(VAL_CHARS) for _ in range(r.randint(2, 6)))


def _apply(layers):
    effective = {}
    for layer in layers:
        for k, action in layer.items():
            if action is None:
                effective.pop(k, None)
            else:
                effective[k] = action
    return effective


def _render_actions(actions):
    parts = []
    for k in sorted(actions.keys()):
        v = actions[k]
        if v is None:
            parts.append("%s = deleted" % k)
        else:
            parts.append("%s = %s" % (k, v))
    return ", ".join(parts)


def _parse(answer):
    result = {}
    for token in answer.split(", "):
        if not token:
            continue
        if token.endswith(" = deleted"):
            result[token[:-len(" = deleted")]] = None
        else:
            k, v = token.split(" = ")
            result[k] = v
    return result


@dataclass
class ConfigPrecedenceResolutionConfig(Config):
    n_keys: int = 3

    def apply_difficulty(self, level):
        self.n_keys = 3 + level


class ConfigurationPrecedenceResolution(Task):
    summary = ("Merge defaults, inherited settings, environment overrides, local "
               "overrides, and deletion markers to return one effective "
               "configuration, with varied key counts and per-layer setting "
               "sparsity.")
    config_cls = ConfigPrecedenceResolutionConfig

    def generate_entry(self):
        cfg = self.config
        n_keys = cfg.n_keys

        keys = []
        while len(keys) < n_keys:
            k = _rand_key(random)
            if k not in keys:
                keys.append(k)

        layers = [dict() for _ in LAYER_NAMES]
        for k in keys:
            for layer in layers:
                flag = random.random()
                if flag < 0.2:
                    layer[k] = None
                elif flag < 0.8:
                    layer[k] = _rand_val(random)

        champion = random.choice(keys)
        layers[-1][champion] = _rand_val(random)

        effective = _apply(layers)
        assert champion in effective

        meta_layers = [
            {"name": name, "settings": _render_actions(layer)}
            for name, layer in zip(LAYER_NAMES, layers)
        ]
        metadata = edict(payload={"layers": meta_layers})
        answer = _render_actions(effective)

        assert _parse(answer) == effective
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = ["%s settings: %s" % (layer["name"], layer["settings"])
                 for layer in metadata.payload["layers"]]
        return (
            "A program reads configuration from four sources, applied in order "
            "from weakest to strongest: defaults, inherited, environment, local. "
            "Later sources override earlier ones for the same key. A setting "
            "marked `deleted` removes that key from the effective configuration "
            "entirely; a later source may re-add it. Given:\n"
            + "\n".join(lines) +
            "\n\nWrite the effective configuration as comma-separated entries "
            "`key = value`, sorted alphabetically by key, omitting any key that "
            "ends up deleted. For a working config with keys `host` and `port` "
            "the answer format is: host = x, port = 123"
        )

    def score_answer(self, answer, entry):
        try:
            got = _parse(answer.strip())
        except Exception:
            return 0.0
        expected = _parse(entry.answer)
        if len(got) != len(expected):
            return 0.0
        for k, v in got.items():
            if expected.get(k) != v:
                return 0.0
        return 1.0
