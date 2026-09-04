from dataclasses import dataclass
import random
import ast

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'hierarchical_rollup (draw 1 of 1)',
 'hypothesis': 'HV-039',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/hierarchical_rollup',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 502759112,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class HierarchicalRollupConfig(Config):
    n_nodes: int = 5
    max_depth: int = 3
    val_range: int = 10

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 2 * level)
        self.max_depth = sround(self.max_depth + level)
        self.val_range = sround(self.val_range + 4 * level)


def _build_tree(n_nodes, max_depth):
    parent = {"A": None}
    depth = {"A": 0}
    children = {"A": []}
    node = 1
    frontier = ["A"]
    attempts = 0
    while node < n_nodes and frontier and attempts < 1000:
        attempts += 1
        nxt = []
        for p in frontier:
            if depth[p] >= max_depth - 1:
                continue
            remaining = n_nodes - node
            k = random.randint(0, min(2, remaining))
            for _ in range(k):
                if node >= n_nodes:
                    break
                nm = chr(ord("A") + node)
                parent[nm] = p
                depth[nm] = depth[p] + 1
                children[p].append(nm)
                children[nm] = []
                node += 1
                nxt.append(nm)
        frontier = nxt
    # guarantee a chain to reach n_nodes if the random growth stalled
    while node < n_nodes:
        # append a leaf to the current deepest node
        deepest = max(depth, key=depth.get)
        nm = chr(ord("A") + node)
        parent[nm] = deepest
        depth[nm] = depth[deepest] + 1
        children[deepest].append(nm)
        children[nm] = []
        node += 1
    return parent, children


def _rollup(root, vals, overrides, children):
    def rec(v):
        cvals = [rec(c) for c in children.get(v, [])]
        base = vals.get(v, 0) + sum(cvals)
        if v in overrides:
            return overrides[v]
        return base
    return rec(root)


class HierarchicalRollup(Task):
    summary = "Aggregate exact values through explicit parent-child hierarchies with exclusions or overrides, returning requested subtotals or rolled-up values."
    config_cls = HierarchicalRollupConfig

    def generate_entry(self):
        cfg = self.config
        n_nodes = max(2, int(cfg.n_nodes))
        val_range = max(2, int(cfg.val_range))
        max_depth = max(2, int(cfg.max_depth))

        for _ in range(20):
            parent, children = _build_tree(n_nodes, max_depth)
            names = sorted(children.keys())
            if len(names) < 2:
                continue
            vals = {nm: random.randint(0, val_range) for nm in names}
            mode = random.choice(["override", "subtree"])
            ask_node = random.choice(names)
            overrides = {}
            excl = None
            if mode == "override":
                t = random.choice(names)
                overrides[t] = random.randint(0, val_range)
                answer = _rollup(ask_node, vals, overrides, children)
            else:
                ch = children.get(ask_node, [])
                if ch:
                    excl = random.choice(ch)
                    def rec_excl(v, from_ask):
                        if v == excl:
                            return 0
                        if v == ask_node:
                            chs = [c for c in children.get(v, []) if c != excl]
                        else:
                            chs = children.get(v, [])
                        return vals.get(v, 0) + sum(rec_excl(c, False) for c in chs)
                    answer = rec_excl(ask_node, True)
                else:
                    answer = _rollup(ask_node, vals, overrides, children)
                    mode = "override"
                    overrides[ask_node] = answer
            break
        else:
            raise RuntimeError("unable to generate tree")

        tree_desc = []
        for p in sorted(children.keys()):
            if children[p]:
                tree_desc.append("{} -> {}".format(p, ", ".join(sorted(children[p]))))
        vals_str = ", ".join("{}={}".format(k, vals[k]) for k in names)
        payload_lines = ["Hierarchy: " + "; ".join(tree_desc),
                         "Per-node values, listed for every node: " + vals_str + "."]
        if mode == "override":
            (k, v) = next(iter(overrides.items()))
            payload_lines.append("An override forces the computed subtotal of '{}' to equal {}.".format(k, v))
            question = "What is the rolled-up subtotal (sum of all values in its subtree, with the override applied) of the subtree rooted at '{}'? The answer is an integer."
            question = question.format(ask_node)
        else:
            payload_lines.append("The requested subtotal excludes the entire subtree under child '{}' of '{}'.".format(excl, ask_node))
            question = ("What is the rolled-up subtotal of the subtree rooted at '{}', "
                        "excluding the entire subtree under its child '{}'? "
                        "The answer is an integer.").format(ask_node, excl)

        metadata = edict({
            "nodes": tree_desc,
            "values": dict(vals),
            "overrides": dict(overrides),
            "ask_node": ask_node,
            "excl": excl,
            "mode": mode,
            "answer": int(answer),
        })
        metadata.payload = {
            "lines": payload_lines,
            "question": question,
        }
        return Entry(metadata=metadata, answer=str(int(answer)))

    def render_prompt(self, metadata):
        return render_payload(metadata.payload)

    def score_answer(self, answer, entry):
        s = answer.strip()
        try:
            a = ast.literal_eval(s)
        except Exception:
            return 0.0
        if isinstance(a, bool):
            return 0.0
        if isinstance(a, (int, float)):
            try:
                gold = ast.literal_eval(entry.answer)
            except Exception:
                return 0.0
            if isinstance(gold, (int, float)) and abs(float(a) - float(gold)) < 1e-9:
                return 1.0
        return 0.0
