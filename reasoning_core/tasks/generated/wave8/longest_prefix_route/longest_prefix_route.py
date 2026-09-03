import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'longest_prefix_route (draw 1 of 2)',
 'hypothesis': 'W1-046',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/longest_prefix_route',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1552420125,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _longest_prefix(dest, routes):
    winner = None
    best = -1
    for (pbits, plen, hop) in routes:
        if dest.startswith(pbits) and plen > best:
            best = plen
            winner = hop
    return winner


@dataclass
class LongestPrefixRouteConfig(Config):
    width: int = 10
    n_routes: int = 4
    n_hops: int = 6

    def apply_difficulty(self, level):
        self.width = sround(max(6, self.width + level))
        self.n_routes = sround(self.n_routes + 2 * level)
        self.n_hops = sround(self.n_hops + 2 * level)


class LongestPrefixRoute(Task):
    summary = "Given bit-prefix routes and a destination bitstring, output the selected next hop."
    config_cls = LongestPrefixRouteConfig

    def generate_entry(self):
        cfg = self.config
        width = int(cfg.width)
        n_routes = int(cfg.n_routes)
        n_hops = int(cfg.n_hops)

        hops = sorted(random.sample(range(0, 1000), n_hops))

        routes = []
        used = set()
        for _ in range(n_routes):
            plen = random.randint(0, width)
            pbits = "".join(random.choice("01") for _ in range(plen))
            key = (plen, pbits)
            if key in used:
                continue
            used.add(key)
            routes.append((pbits, plen, random.choice(hops)))

        if not routes:
            routes.append(("", 0, random.choice(hops)))

        if (0, "") not in used:
            routes.append(("", 0, random.choice(hops)))

        dest = "".join(random.choice("01") for _ in range(width))
        answer_hop = _longest_prefix(dest, routes)
        if answer_hop is None:
            raise RuntimeError("no matching route")

        ordered = sorted((plen, pbits, hop) for (pbits, plen, hop) in routes)
        payload_routes = [{"length": plen, "prefix": pbits or "(default)", "hop": hop}
                          for (plen, pbits, hop) in ordered]

        metadata = edict({
            "width": width,
            "routes": [[plen, pbits, hop] for (pbits, plen, hop) in routes],
            "destination": dest,
            "winning_hop": answer_hop,
        })
        metadata.payload = {"destination": dest, "routes": payload_routes}
        return Entry(metadata=metadata, answer=str(answer_hop))

    def render_prompt(self, metadata):
        body = render_payload(metadata.payload)
        return (
            f"{body}\n\n"
            "A route (length, prefix, hop) forwards a destination to hop whenever the first "
            "'length' bits of the destination bitstring equal 'prefix'. A route with length 0 has "
            "prefix \"(default)\" and matches every destination. Apply longest-prefix matching: "
            "among all matching routes, the one with the greatest length wins; if multiple have "
            "the same greatest length the tie is not reached because prefixes of one length are "
            "distinct. The answer is the winning hop, as an integer."
        )

    def score_answer(self, answer, entry):
        try:
            return 1.0 if int(str(answer).strip()) == int(entry.answer) else 0.0
        except Exception:
            return 0.0
