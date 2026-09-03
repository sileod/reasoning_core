"""BGP best-path selection.

Given a set of BGP routes each carrying local-preference, AS-path length,
origin, MED and neighbor router attributes, apply a stated best-path rule
sequence to select the single winning route.
"""

import random

from reasoning_core.template import Entry, Task, Config, edict, render_payload, stochastic_rounding as sround

ORIGIN_ORDER = {"IGP": 0, "EGP": 1, "incomplete": 2}
ORIGINS = ["IGP", "EGP", "incomplete"]
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]


def _select_best(routes):
    """Apply the best-path rule sequence. Returns the winning route dict."""
    candidates = list(routes)
    key = lambda r: (
        -r["local_pref"],
        r["as_path"],
        ORIGIN_ORDER[r["origin"]],
        r["med"],
        r["neighbor"],
    )
    return min(candidates, key=key)


@staticmethod
def _parse_routes(text):
    routes = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        name = parts[0]
        lp = int(parts[1])
        asp = int(parts[2])
        orig = parts[3]
        med = int(parts[4])
        nbr = parts[5]
        routes.append({
            "name": name, "local_pref": lp, "as_path": asp,
            "origin": orig, "med": med, "neighbor": nbr,
        })
    return routes


class BgpBestPathConfig(Config):
    n_shared: int = 2
    n_routes: int = 4
    n_ties: int = 1

    def apply_difficulty(self, level):
        self.n_routes = sround(self.n_routes + level)
        self.n_shared = sround(self.n_shared + (1 if level >= 2 else 0))
        self.n_ties = sround(self.n_ties + (1 if level >= 4 else 0))


class BgpBestPath(Task):
    summary = "Given BGP route attributes, output the route selected by a stated best-path rule sequence."
    config_cls = BgpBestPathConfig
    task_version = 2

    def generate_entry(self):
        n = self.config.n_routes
        names = LETTERS[:n]
        routes = []

        # Generate per-route attributes with wide local-pref to make the winner vary.
        for i in range(n):
            ip_hi = random.randint(1, 200)
            ip_mid = random.randint(1, 200)
            nbr = "10.%d.%d.1" % (ip_hi, ip_mid)
            routes.append({
                "name": names[i],
                "local_pref": random.randint(60, 220),
                "as_path": random.randint(1, 6),
                "origin": random.choice(ORIGINS),
                "med": random.randint(0, 80),
                "neighbor": nbr,
            })

        winner = _select_best(routes)

        display = list(routes)
        random.shuffle(display)

        payload_lines = []
        for r in display:
            payload_lines.append("%s local-pref %d as-path-length %d origin %s med %d neighbor %s" % (
                r["name"], r["local_pref"], r["as_path"], r["origin"], r["med"], r["neighbor"]))
        payload = "\n".join(payload_lines)

        metadata = edict({})
        metadata.routes = routes
        metadata.winner = winner["name"]
        metadata.payload = {"routes": payload}
        metadata.n_routes = n

        answer = winner["name"]
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        header = (
            "A router must select the best path to a destination from the candidate "
            "routes below. The best-path rule sequence, applied in order, is:\n"
            "1. Highest local-preference wins.\n"
            "2. If tied, the shortest AS-path length wins.\n"
            "3. If still tied, the lowest origin value wins (IGP=0, EGP=1, incomplete=2).\n"
            "4. If still tied, the lowest MED wins.\n"
            "5. If still tied, the lowest neighbor IP address wins.\n\n"
        )
        return header + render_payload(metadata.payload) + (
            "\n\nThe champion route is the single route selected by that rule sequence. "
            "Print its identifier letter, e.g. A."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip()
        if a == entry.answer:
            return 1.0
        if len(a) == 1 and a.upper() == entry.answer:
            return 1.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'bgp_best_path (draw 2 of 2)',
 'hypothesis': 'W1-051',
 'changes': 'new task in reasoning_core/tasks/generated/wave8/bgp_best_path',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3767003075,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
