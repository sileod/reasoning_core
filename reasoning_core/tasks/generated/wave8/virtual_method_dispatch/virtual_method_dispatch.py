"""Virtual method dispatch: given a class hierarchy with method overrides, report which
definition is actually invoked for a method call at runtime."""

import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'virtual_method_dispatch (draw 1 of 2)',
 'hypothesis': 'W1-059',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/virtual_method_dispatch',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 480048984,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


METHOD_NAMES = ["describe", "compute", "render", "parse", "execute", "fetch",
                "transform", "serialize", "load", "handle"]


class VirtualMethodDispatchConfig(Config):
    n_classes: int = 5
    n_methods: int = 2

    def apply_difficulty(self, level):
        self.n_classes = sround(self.n_classes + level)
        self.n_methods = sround(self.n_methods + level // 2)


def _render_class_definitions(hierarchy, overrides, root):
    lines = []
    all_cls = set(hierarchy.keys()) | set(hierarchy.values())
    ordered = sorted(all_cls)
    methods = sorted({name for name, _c, _b in overrides})
    # ensure root has methods if lacking
    for cls in ordered:
        if cls == root:
            par = "object"
        else:
            par = hierarchy[cls]
        lines.append(f"class {cls}({par}):")
        cls_defs = [(name, body) for (name, c2, body) in overrides if c2 == cls]
        if not cls_defs:
            lines.append("    pass")
        else:
            for name, body in sorted(cls_defs):
                indented = body.replace("\n", "\n    ")
                lines.append(f"    {indented}")
    return "\n".join(lines)


class VirtualMethodDispatch(Task):
    summary = ("Given a class hierarchy with method overrides and a runtime class, "
               "determine which method definition is invoked by virtual dispatch, "
               "covering chains of redefined methods and inherited fallbacks.")
    config_cls = VirtualMethodDispatchConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        # Build a class hierarchy: a DAG where classes (except root) have one parent.
        n = max(3, cfg.n_classes)
        n_methods = max(1, cfg.n_methods)
        names = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]

        root = "A"
        hier = {}
        used = [root]
        # deterministic ordering
        remaining = sorted(names[: n - 1])
        for i, c in enumerate(remaining):
            # pick parent among already created (closer to a chain with some branching)
            parent = used[random.randrange(len(used))]
            hier[c] = parent
            used.append(c)

        all_cls = set(hier.keys()) | set(hier.values())
        method_names = METHOD_NAMES[:n_methods]

        # Decide overrides. Each class (except root) may override some methods.
        overrides = []
        bodies = {}
        for m in method_names:
            # root defines base for the method
            bodies[(root, m)] = f"def {m}(self):\n        return '{m}(A)'"
            # each non-root class overrides with probability ~0.7
            for c in sorted(all_cls):
                if c == root:
                    continue
                if random.random() < 0.65:
                    bodies[(c, m)] = f"def {m}(self):\n        return '{m}({c})'"

        for (cls, m), body in sorted(bodies.items()):
            overrides.append((m, cls, body))

        # Choose a runtime class and a method to dispatch.
        runtime = random.choice(sorted(all_cls))
        method = random.choice(method_names)

        # Resolve: walk up from runtime to root, first class that defines method.
        def resolve(rcls, m):
            cur = rcls
            while True:
                if (cur, m) in bodies:
                    return cur, bodies[(cur, m)]
                if cur == root:
                    return root, bodies[(root, m)]
                cur = hier[cur]

        def_cls, def_body = resolve(runtime, method)
        answer = def_body.strip().split("\n")[0]  # the def line

        # Express answer as "Class X: def method(self)" entry point
        # The answer we score is the defining class name.
        hierarchy_repr = dict(hier)
        metadata = edict({
            "hierarchy": hierarchy_repr,
            "overrides": [[m, c, b] for (m, c, b) in overrides],
            "root": root,
            "runtime": runtime,
            "method": method,
            "answer_class": def_cls,
        })
        metadata.payload = {
            "classes": _render_class_definitions(hier, overrides, root),
            "runtime": runtime,
            "method": method,
        }
        return Entry(metadata=metadata, answer=str(def_cls))

    def render_prompt(self, metadata):
        return (
            f"{metadata.payload['classes']}\n\n"
            f"A program holds an object whose runtime class is "
            f"{metadata.payload['runtime']} and calls the method "
            f"{metadata.payload['method']}(). Virtual dispatch resolves this to the "
            f"most-derived definition in the hierarchy. "
            f"Which class provides the definition that is actually invoked? "
            f"Answer with the class name only."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        gold = entry.answer
        if answer.strip() == gold:
            return 1.0
        return 0.0


def _resolve_for_test(hier, overrides, runtime, method):
    bodies = {}
    root = [c for c in (set(hier.keys()) | set(hier.values())) if c not in hier][0]
    for m, cls, body in overrides:
        bodies[(cls, m)] = body
    for m, cls, body in overrides:
        if (root, m) not in bodies and cls == root:
            bodies[(root, m)] = body
    cur = runtime
    while True:
        if (cur, method) in bodies:
            return cur
        if cur == root:
            return root
        if cur not in hier:
            return root
        cur = hier[cur]
