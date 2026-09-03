import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class LexicalScopeResolutionV1Config(Config):
    n_scopes: int = 3
    n_decls: int = 4

    def apply_difficulty(self, level):
        self.n_scopes = sround(self.n_scopes + level)
        self.n_decls = sround(self.n_decls + level)


NAME_POOL = [
    "x", "y", "z", "w", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "m", "n", "p", "q", "r",
    "u", "v", "t", "s", "o", "l",
]

_DECL_PREFIX = "decl"
_DECL_BASE = 100


def _render_scopes(scopes):
    lines = []
    for scope in scopes:
        if scope["parent"] is None:
            lines.append("scope(root):")
        else:
            lines.append("scope(inside scope %d):" % scope["id"])
        for depth, (name, val) in enumerate(scope["decls"]):
            indent = "  "
            lines.append("%s%s = %s" % (indent, name, val))
    return "\n".join(lines)


def build_instance(n_scopes, n_decls, rng):
    while True:
        table = {}
        scopes = []

        for i in range(n_scopes):
            parent = i - 1 if i > 0 else None
            if parent is not None and i >= 2 and rng.random() < 0.4:
                parent = rng.randint(0, i - 2)
            n_here = rng.randint(1, max(2, n_decls))
            names = rng.sample(NAME_POOL, min(len(NAME_POOL), n_here))
            decls = []
            for name in names:
                val = _DECL_BASE + len(table)
                table[(i, name)] = val
                decls.append((name, val))
            scopes.append({"id": i, "parent": parent, "decls": decls})

        scopes_sorted = sorted(scopes, key=lambda s: s["id"])

        query_scope = scopes_sorted[-1]
        candidate_names = [name for _, (name, _val) in enumerate(query_scope["decls"])]
        if not candidate_names:
            continue

        name = rng.choice(candidate_names)

        resolved_val = None
        cur = query_scope
        seen = set()
        while cur is not None and cur["id"] not in seen:
            seen.add(cur["id"])
            found = None
            for dn, dv in cur["decls"]:
                if dn == name:
                    found = dv
                    break
            if found is not None:
                resolved_val = found
                break
            cur = next((s for s in scopes_sorted if s["id"] == cur["parent"]), None)

        if resolved_val is None:
            continue

        answer = "%s = %d" % (name, resolved_val)
        return scopes_sorted, query_scope, name, answer


class LexicalScopeResolution(Task):
    summary = "Given nested scopes and declarations, output the declaration bound to a queried identifier occurrence."
    config_cls = LexicalScopeResolutionV1Config

    def generate_entry(self, level=None):
        cfg = self.config
        n_scopes = cfg.n_scopes
        n_decls = cfg.n_decls
        scopes, query_scope, name, answer = build_instance(n_scopes, n_decls, random)
        metadata = edict({
            "scopes": [[(n, int(v)) for (n, v) in s["decls"]] for s in scopes],
            "query_scope": query_scope["id"],
            "query_name": name,
        })
        metadata.payload = {
            "program": _render_scopes(scopes),
            "query": "What does %s resolve to inside scope %d?" % (name, query_scope["id"]),
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = metadata.payload
        return (
            "Consider the following program with nested scopes. Inner scopes see "
            "declarations from their enclosing scopes, and a name binds to the nearest "
            "declaration in the lexical scope chain (lexical scoping).\n\n"
            "%s\n\n"
            "%s\n\n"
            "Give the answer as the binding in the form 'name = value'." % (
                render_payload({"program": payload["program"]}),
                payload["query"],
            )
        )

    def score_answer(self, answer, entry):
        gt = entry.answer
        if not isinstance(answer, str):
            return 0.0
        return 1.0 if answer.strip() == gt else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'lexical_scope_resolution (draw 1 of 2)',
 'hypothesis': 'W1-053',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/lexical_scope_resolution',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3759181579,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
