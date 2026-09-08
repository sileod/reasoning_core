import ast
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config


@dataclass
class CounterfactualReplayConfig(Config):
    n_layers: int = 3
    n_per_layer: int = 3
    w_range: int = 3
    off_range: int = 10

    def apply_difficulty(self, level):
        self.n_layers = 2 + level
        self.n_per_layer = 2 + (level * 2) // 3
        self.w_range = 2 + level // 2
        self.off_range = 9 + 6 * level


def _build(cfg):
    """Build a layered affine DAG of named integer quantities.

    Layer 0 index 0 is the changed source `c`. Node 0 of every later layer is a
    single-parent chain from `c` (a descendant). Nodes 1..n_per-1 of every layer
    form an independent region that never references `c`. The last layer's node 0
    (the target) sums the chain end plus one independent node, so it depends on
    both the changed event and a retained independent event.
    """
    n_layers = cfg.n_layers
    n_per = cfg.n_per_layer
    layer0 = [random.randint(-cfg.off_range, cfg.off_range) for _ in range(n_per)]
    c_idx = 0
    c_name = "q%d" % c_idx
    layers = [layer0]
    parents = []
    desc = [{c_idx}]
    name_per_layer = [["q%d" % i for i in range(n_per)]]
    counter = n_per
    for l in range(1, n_layers):
        prev = layers[-1]
        pdesc = desc[-1]
        plist = []
        row = []
        layer_names = []
        w0 = random.choice([-1, 1]) * random.randint(1, cfg.w_range)
        off0 = random.randint(-cfg.off_range, cfg.off_range)
        par0 = [0]
        ws0 = [w0]
        if l == n_layers - 1 and n_per >= 2:
            ip = random.randint(1, n_per - 1)
            par0 = [0, ip]
            ws0 = [w0, random.choice([-1, 1]) * random.randint(1, cfg.w_range)]
        val0 = sum(ws0[i] * prev[par0[i]] for i in range(len(par0))) + off0
        row.append(val0)
        plist.append((par0, ws0, off0))
        layer_names.append("q%d" % counter)
        counter += 1
        for j in range(1, n_per):
            k = random.randint(1, min(3, n_per - 1))
            par = sorted(random.sample(range(1, n_per), k))
            ws = [random.choice([-1, 1]) * random.randint(1, cfg.w_range) for _ in range(k)]
            off = random.randint(-cfg.off_range, cfg.off_range)
            val = sum(ws[i] * prev[par[i]] for i in range(k)) + off
            row.append(val)
            plist.append((par, ws, off))
            layer_names.append("q%d" % counter)
            counter += 1
        rows_d = {j for j in range(n_per)
                  if any(p in pdesc for p in plist[j][0])}
        layers.append(row)
        parents.append(plist)
        desc.append(rows_d)
        name_per_layer.append(layer_names)
    return layers, parents, desc, c_idx, c_name, name_per_layer


def _replay(layers, parents, desc, c_idx, c_new):
    """Replay after setting the changed source to c_new: recompute descendants,
    retain non-descendants at their original values."""
    rep = [layers[0][:]]
    rep[0][c_idx] = c_new
    for l in range(1, len(layers)):
        plist = parents[l - 1]
        prev_rep = rep[-1]
        row = []
        for j in range(len(plist)):
            if j in desc[l]:
                par, ws, off = plist[j]
                row.append(sum(ws[i] * prev_rep[par[i]] for i in range(len(par))) + off)
            else:
                row.append(layers[l][j])
        rep.append(row)
    return rep


def _expr(par, ws, off, names):
    terms = []
    for i, w in enumerate(ws):
        nm = names[par[i]]
        if w == 1:
            terms.append(nm)
        elif w == -1:
            terms.append("-" + nm)
        else:
            terms.append("%d*%s" % (w, nm))
    s = " + ".join(terms) if terms else "0"
    s = s.replace("+ -", "- ")
    if off != 0:
        s = s + (" + %d" % off if off > 0 else " - %d" % abs(off))
    return s


class CounterfactualReplay(Task):
    summary = ("Predict one outcome after changing an earlier event, recomputing downstream "
               "effects while retaining independent events, over layered affine integer "
               "dependency chains that mix a changed branch with a retained independent branch.")
    config_cls = CounterfactualReplayConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(200):
            if cfg.n_layers < 2 or cfg.n_per_layer < 2:
                raise RuntimeError("counterfactual_replay: config too small")
            layers, parents, desc, c_idx, c_name, name_per_layer = _build(cfg)
            last = len(layers) - 1
            # The target is last layer node 0 (chain end) whenever it is a valid
            # descendant that also has a retained independent parent.
            par, ws, off = parents[last - 1][0]
            if not (0 in desc[last] and any(p not in desc[last - 1] for p in par)):
                continue

            c_orig = layers[0][c_idx]
            c_new = c_orig
            while c_new == c_orig:
                c_new = random.randint(-cfg.off_range - 4, cfg.off_range + 4)
            rep = _replay(layers, parents, desc, c_idx, c_new)
            orig_t = layers[last][0]
            ans = rep[last][0]
            if ans == orig_t:
                continue

            lines = []
            for ll in range(len(layers)):
                for j in range(len(layers[ll])):
                    nm = name_per_layer[ll][j]
                    if ll == 0:
                        lines.append("%s = %d" % (nm, layers[0][j]))
                    else:
                        ppar, wws, ooff = parents[ll - 1][j]
                        lines.append("%s = %s" % (nm, _expr(ppar, wws, ooff, name_per_layer[ll - 1])))

            t_name = name_per_layer[last][0]
            assert rep[last][0] == ans
            ans_i = int(ans)
            metadata = {
                "lines": lines,
                "changed": c_name,
                "c_orig": int(c_orig),
                "c_new": int(c_new),
                "target": t_name,
                "answer": ans_i,
                "payload": {
                    "lines": lines,
                    "changed": c_name,
                    "c_orig": int(c_orig),
                    "c_new": int(c_new),
                    "target": t_name,
                },
            }
            return Entry(metadata=metadata, answer=str(ans_i))
        raise RuntimeError("counterfactual_replay: could not build a valid instance")

    def render_prompt(self, metadata):
        body = "\n".join(metadata.payload["lines"])
        ch = metadata.payload["changed"]
        tg = metadata.payload["target"]
        co = metadata.payload["c_orig"]
        cn = metadata.payload["c_new"]
        return (
            "Consider these quantities, computed in order so that each later quantity uses "
            "only earlier ones:\n"
            f"{body}\n"
            f"In the actual scenario {ch} equals {co}.\n"
            f"Now suppose that instead {ch} had been {cn} (instead of {co}). Recompute every "
            f"quantity that depends, directly or indirectly, on {ch} using this new value, and "
            f"keep every quantity that does not depend on {ch} exactly as it originally was. "
            f"What is the resulting value of {tg}?\n"
            f"The answer is the single integer value of {tg} (it may be negative)."
        )

    def score_answer(self, answer, entry):
        gt = int(entry.answer)
        s = answer.strip()
        try:
            a = int(ast.literal_eval(s))
        except Exception:
            try:
                a = int(s)
            except Exception:
                return 0.0
        return 1.0 if a == gt else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'counterfactual_replay (draw 1 of 2)',
 'hypothesis': 'ASTRA0-16',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/counterfactual_replay',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2928563038,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
