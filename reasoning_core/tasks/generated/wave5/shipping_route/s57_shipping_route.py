import heapq
import random
import string

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'Add cheapest-route reasoning where the answer is the route, not its '
         'cost.',
 'hypothesis': 'S57',
 'changes': 'Ask which sequence of legs a shipment takes under a stated '
            'tie-break.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3982785801,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _place_names(n):
    prefixes = ["North", "South", "East", "West", "New", "Old", "Port", "Fort",
                "Saint", "Lake", "Mount", "River", "Cape", "Bay", "Far", "Deep",
                "High", "Low", "Bridge", "Stone", "Green", "Red", "White",
                "Black", "Silver", "Golden", "Iron", "Maple", "Pine", "Cedar",
                "Buck", "Dover", "Har", "Fal", "Lin", "Ash", "Elm", "Oak",
                "Brae", "Craig", "Glen", "Strand", "Wick", "Bur", "Cald",
                "Der", "Ferm"]
    suffixes = ["ton", "bury", "mouth", "ford", "field", "ham", "hill",
                "borough", "haven", "port", "stead", "wick", "sea", "mere",
                "ness", "bridge", "dale", "gate", "holme", "ley", "low",
                "stock", "worth", "combe", "cote", "den", "ey", "fleet",
                "hurst", "ington", "ley", "rock", "shaw", "stead", "throp",
                "well", "worth", "by", "gate", "side", "pool", "spring"]
    names = set()
    while len(names) < n:
        p = random.choice(prefixes)
        s = random.choice(suffixes)
        names.add(p + s)
    return sorted(names)[:n]


class ShippingRouteConfig(Config):
    n_places: int = 6
    n_legs: int = 8
    max_cost: int = 12

    def apply_difficulty(self, level):
        self.n_places = int(self.n_places + level * 2)
        self.n_legs = int(self.n_legs + level * 3)
        self.max_cost = int(self.max_cost + level * 3)


def _solve(start, target, graph):
    INF = float("inf")
    dist = {v: INF for v in graph}
    legs = {v: INF for v in graph}
    parent = {v: None for v in graph}
    dist[start] = 0
    legs[start] = 0
    pq = [(0, 0, start)]
    while pq:
        d, l, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, c in graph[u]:
            nd = d + c
            nl = l + 1
            if nd < dist[v]:
                dist[v] = nd
                legs[v] = nl
                parent[v] = u
                heapq.heappush(pq, (nd, nl, v))
            elif nd == dist[v] and nl < legs[v]:
                legs[v] = nl
                parent[v] = u
            elif nd == dist[v] and nl == legs[v] and u < v:
                parent[v] = u
    if dist[target] == INF:
        return None
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


class ShippingRoute(Task):
    config_cls = ShippingRouteConfig

    def generate_entry(self):
        n_places = self.config.n_places
        n_legs = self.config.n_legs
        max_cost = self.config.max_cost

        names = _place_names(n_places)

        max_direct = max(3, max_cost)
        while True:
            legs = []
            leg_set = set()
            for _ in range(n_legs):
                a, b = random.sample(range(n_places), 2)
                if a > b:
                    a, b = b, a
                c = random.randint(1, max_cost)
                legs.append((a, b, c))
                leg_set.add((a, b))

            graph = {i: [] for i in range(n_places)}
            for a, b, c in legs:
                graph[a].append((b, c))
                graph[b].append((a, c))

            start, target = random.sample(range(n_places), 2)

            direct_cost = None
            if (start, target) in leg_set or (target, start) in leg_set:
                for a, b, c in legs:
                    if set((a, b)) == set((start, target)):
                        direct_cost = c
                        break

            path = _solve(start, target, graph)
            if path is None:
                continue

            path_cost = 0
            for i in range(len(path) - 1):
                for a, b, c in legs:
                    if set((a, b)) == set((path[i], path[i + 1])):
                        path_cost += c
                        break

            chosen = None
            if direct_cost is not None:
                direct_path = [start, target]
                if (path_cost < direct_cost) or (
                        path_cost == direct_cost and len(path) < len(direct_path)):
                    chosen = path
                else:
                    chosen = direct_path
            else:
                chosen = path

            if len(chosen) > 2:
                break

        place = lambda i: names[i]
        leg_lines = []
        for a, b, c in legs:
            leg_lines.append("The direct leg between %s and %s costs %d." % (
                place(a), place(b), c))

        route_str = " -> ".join(place(i) for i in chosen)
        metadata = edict({
            "places": [place(i) for i in range(n_places)],
            "legs": [(place(a), place(b), c) for a, b, c in legs],
            "start": place(start),
            "target": place(target),
            "route": route_str,
        })
        metadata.payload = {
            "places": metadata.places,
            "legs": metadata.legs,
            "start": metadata.start,
            "target": metadata.target,
        }
        answer = route_str
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        lines = ["A network of towns follows, with the cost of each direct "
                 "stage given."]
        lines.append("Towns: " + ", ".join(p["places"]) + ".")
        for a, b, c in p["legs"]:
            lines.append("%s <-> %s costs %d." % (a, b, c))
        lines.append("A shipment must go from %s to %s." % (p["start"], p["target"]))
        lines.append(
            "Give the cheapest route as an arrow-separated list of town names "
            "that the shipment takes, for example \"A -> B -> C\". Among "
            "equally cheap routes the one with fewer legs wins, and if the "
            "leg count is also tied the alphabetically smaller sequence of "
            "town names wins. The answer is the route, exactly as described.")
        if p["start"] == p["target"]:
            return "\n".join(lines) + "\n\nThe answer is the town itself."
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        gold = entry.answer
        if not isinstance(answer, str):
            return 0.0
        a = "".join(answer.split())
        g = "".join(gold.split())
        return 1.0 if a == g else 0.0
