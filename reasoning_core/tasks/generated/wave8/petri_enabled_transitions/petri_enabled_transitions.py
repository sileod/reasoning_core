import ast
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'petri_enabled_transitions (draw 1 of 2)',
 'hypothesis': 'W1-077',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/petri_enabled_transitions',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1701447588,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _enabled(place_tokens, pre, all_trans):
    """Return sorted tuple of transitions that are enabled.

    place_tokens: dict place->tokens
    pre: dict trans -> tuple of (place, weight)
    conns: set of (trans, place) present in pre
    all_trans: list of transition labels
    """
    enabled = []
    for t in all_trans:
        if t in pre:
            ok = True
            for (p, w) in pre[t]:
                if place_tokens.get(p, 0) < w:
                    ok = False
                    break
            if ok:
                enabled.append(t)
    return tuple(sorted(enabled))


def _format_enabled(tup):
    if not tup:
        return "none"
    return ", ".join(tup)


def _parse_enabled(text):
    if text is None:
        return None
    s = text.strip().lower()
    if not s:
        return None
    if s == "none":
        return ()
    parts = [x.strip().strip('"\'') for x in s.split(",")]
    parts = [p for p in parts if p]
    return tuple(sorted(parts))


class PetriEnabledConfig(Config):
    n_places: int = 4
    n_trans: int = 5
    max_tokens: int = 6
    max_weight: int = 3

    def apply_difficulty(self, level):
        self.n_places = sround(self.n_places + level)
        self.n_trans = sround(self.n_trans + level)
        self.max_tokens = sround(self.max_tokens + level)
        self.max_weight = sround(self.max_weight + level)


class PetriEnabledTransitions(Task):
    summary = ("Given a Petri-net marking and weighted pre-arcs, name all currently "
               "enabled transitions in sorted order, with an empty result reported as "
               "'none'.")
    config_cls = PetriEnabledConfig

    def generate_entry(self):
        cfg = self.config
        n_places = int(cfg.n_places)
        n_trans = int(cfg.n_trans)
        max_tokens = int(cfg.max_tokens)
        max_weight = int(cfg.max_weight)

        places = [f"p{i}" for i in range(n_places)]
        trans = [f"t{i}" for i in range(n_trans)]

        for _attempt in range(400):
            tokens = {}
            for p in places:
                tokens[p] = random.randint(0, max_tokens)

            pre = {}
            for t in trans:
                k = random.randint(1, min(3, n_places))
                chosen = random.sample(places, k)
                edges = []
                for p in chosen:
                    w = random.randint(1, max_weight)
                    edges.append((p, w))
                pre[t] = tuple(edges)

            enabled = _enabled(tokens, pre, trans)
            gold = _format_enabled(enabled)
            if not gold:
                continue
            break
        else:
            tokens[random.choice(places)] = max_tokens
            pre = {t: tuple() for t in trans}
            enabled = tuple(sorted(trans))
            gold = _format_enabled(enabled)

        payload = {
            "places": {p: tokens[p] for p in places},
            "transitions": list(trans),
            "pre_arcs": sorted(
                [(t, p, w) for t in trans for (p, w) in pre.get(t, ())],
                key=lambda x: (int(x[0][1:]), int(x[1][1:])),
            ),
        }

        metadata = edict({
            "places": payload["places"],
            "pre_arcs": payload["pre_arcs"],
            "_enabled": enabled,
        })
        metadata.payload = payload
        return Entry(metadata=metadata, answer=gold)

    def render_prompt(self, metadata):
        return (
            "A Petri net has places with the given token counts and weighted pre-arcs "
            "from each transition to its input places. A transition is enabled when "
            "every input place holds at least the arc weight of tokens. "
            "Given the marking and arcs, list all enabled transitions in the answer, "
            "in lexicographic (sorted) order, separated by commas. If no transition "
            "is enabled, the answer is exactly 'none'.\n\n"
            f"{render_payload(metadata.payload)}\n\nThe answer is:"
        )

    def score_answer(self, answer, entry):
        parsed = _parse_enabled(answer)
        if parsed is None:
            return 0.0
        gold = _parse_enabled(entry.answer)
        return 1.0 if parsed == gold else 0.0
