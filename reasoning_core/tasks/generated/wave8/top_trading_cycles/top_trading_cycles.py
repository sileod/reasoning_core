"""Top Trading Cycles: given agents, owned items, and strict preferences,
output the item assigned to a queried agent."""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'top_trading_cycles (draw 1 of 2)',
 'hypothesis': 'W1-066',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/top_trading_cycles',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 623615645,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _run_ttc(n, prefs):
    """prefs[i] is a list of item ids (0..n-1) most->least preferred by agent i.

    Return dict mapping agent id -> item id it is assigned (TTC matching)."""
    remaining_agents = set(range(n))
    remaining_items = set(range(n))
    match = {}
    while remaining_agents:
        # each remaining agent points to its most-preferred remaining item
        point = {}
        for a in remaining_agents:
            for it in prefs[a]:
                if it in remaining_items:
                    point[a] = it
                    break
        # each remaining item points to its owner
        owner = {}
        for it in remaining_items:
            owner[it] = it  # item i owned by agent i
        # find a cycle in the agent->item->agent ... graph
        # follow from an arbitrary remaining agent
        start = next(iter(remaining_agents))
        seen = {}
        cur = start
        chain = []
        while cur not in seen:
            seen[cur] = len(chain)
            chain.append(cur)
            it = point[cur]
            cur = owner[it]
        cycle_start_idx = seen[cur]
        cycle_agents = chain[cycle_start_idx:]
        for a in cycle_agents:
            match[a] = point[a]
            remaining_agents.discard(a)
            remaining_items.discard(point[a])
    return match


@dataclass
class TopTradingCyclesConfig(Config):
    n_agents: int = 5

    def apply_difficulty(self, level):
        self.n_agents = sround(self.n_agents + level)


def _letter(i):
    return chr(ord("A") + i)


class TopTradingCycles(Task):
    summary = ("Given agents with their owned single items and strict preference "
               "lists, run the Top Trading Cycles algorithm and output the item "
               "assigned to a queried agent across varied market sizes.")
    config_cls = TopTradingCyclesConfig

    def generate_entry(self):
        n = int(self.config.n_agents)
        # random strict preference for each agent over all items
        prefs = []
        for _ in range(n):
            p = list(range(n))
            random.shuffle(p)
            prefs.append(tuple(p))
        # TTC matching is unique; verify it is a bijection on all agents/items.
        match = _run_ttc(n, prefs)
        assert len(match) == n
        assert sorted(match.values()) == list(range(n))

        q = random.randrange(n)
        answer_item = match[q]
        answer = _letter(answer_item)

        payload = {
            "agents": [
                {
                    "name": str(i + 1),
                    "owns": _letter(i),
                    "preferences": [_letter(x) for x in prefs[i]],
                }
                for i in range(n)
            ],
            "query": {
                "agent": str(q + 1),
            },
        }
        metadata = edict({"payload": payload})
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = []
        for ag in metadata.payload["agents"]:
            lines.append(
                "Agent %s owns item %s and ranks the items from most to least "
                "preferred as: %s."
                % (
                    ag["name"],
                    ag["owns"],
                    ", ".join(ag["preferences"]),
                )
            )
        body = "\n".join(lines)
        q = metadata.payload["query"]["agent"]
        return (
            f"Each of the following agents owns one distinct item, and every "
            f"agent has a strict preference ranking of all the items (most "
            f"preferred first). Using the Top Trading Cycles algorithm, each "
            f"agent points to their most preferred remaining item, every item "
            f"points to its owner, cycles are found and their members receive "
            f"the item they point to, and the members of each cycle are removed "
            f"along with their items; this repeats until every agent has an item."
            f"\n\n{body}\n\n"
            f"After applying Top Trading Cycles, which item does Agent {q} end "
            f"up assigned? "
            f"\n\nThe answer is a single letter naming that item."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        return 1.0 if answer.strip().upper() == entry.answer.upper() else 0.0
