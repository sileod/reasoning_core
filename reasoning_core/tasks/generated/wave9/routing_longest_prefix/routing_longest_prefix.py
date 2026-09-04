import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'routing_longest_prefix (draw 1 of 1)',
 'hypothesis': 'HV-047',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/routing_longest_prefix',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 265960371,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _prefix_mask(bits, plen):
    if plen == 0:
        return 0
    return ((1 << plen) - 1) << (bits - plen)


def _matches(bits, prefix, plen, addr):
    return (addr & _prefix_mask(bits, plen)) == (prefix & _prefix_mask(bits, plen))


def _select_hop(bits, addr, entries):
    best_plen = -1
    best_metric = None
    best_hop = None
    for e in entries:
        if not _matches(bits, e["prefix"], e["plen"], addr):
            continue
        if e["plen"] > best_plen:
            best_plen = e["plen"]
            best_metric = e["metric"]
            best_hop = e["hop"]
        elif e["plen"] == best_plen:
            if e["metric"] < best_metric:
                best_metric = e["metric"]
                best_hop = e["hop"]
            elif e["metric"] == best_metric and e["hop"] < best_hop:
                best_hop = e["hop"]
    return best_hop


def _dst(addr, bits):
    return format(addr, "0{}b".format(bits))


def _fmt_prefix(bits, prefix, plen):
    return _dst(prefix, bits)[:plen]


@dataclass
class RoutingLongestPrefixConfig(Config):
    bits: int = 9
    n_entries: int = 8
    n_hops: int = 9
    max_metric: int = 16

    def apply_difficulty(self, level):
        self.bits = sround(self.bits + 0.7 * level)
        self.n_entries = sround(self.n_entries + 1.1 * level)
        self.n_hops = sround(self.n_hops + 1.0 * level)
        self.max_metric = sround(self.max_metric + 2.0 * level)


class RoutingLongestPrefix(Task):
    summary = "Apply longest-prefix routing with metric and tie rules to binary or IP-like destination prefixes, returning the selected next hop."
    config_cls = RoutingLongestPrefixConfig

    def generate_entry(self):
        bits = max(5, min(22, int(self.config.bits)))
        n_entries = max(4, min(18, int(self.config.n_entries)))
        n_hops = max(4, min(n_entries, int(self.config.n_hops)))
        max_metric = max(2, int(self.config.max_metric))

        addr = random.randint(0, (1 << bits) - 1)

        hops = ["R{}".format(i + 1) for i in range(n_hops)]

        best = None
        for _ in range(400):
            n_match = random.randint(2, min(n_entries, n_hops))
            entries = []
            chosen = random.sample(hops, n_match)
            for i in range(n_match):
                plen = random.randint(1, bits)
                prefix = (addr >> (bits - plen)) << (bits - plen) if plen > 0 else 0
                entries.append({
                    "prefix": prefix,
                    "plen": plen,
                    "hop": chosen[i],
                    "metric": random.randint(0, max_metric),
                })
            for j in range(n_match, n_entries):
                plen = random.randint(1, bits)
                prefix = ((addr ^ (1 << (bits - 1))) >> (bits - plen)) << (bits - plen)
                entries.append({
                    "prefix": prefix,
                    "plen": plen,
                    "hop": random.choice(hops),
                    "metric": random.randint(0, max_metric),
                })
            winner = _select_hop(bits, addr, entries)
            if winner is None:
                continue
            again = _select_hop(bits, addr, entries)
            if again != winner:
                continue
            if not any(_matches(bits, e["prefix"], e["plen"], addr) for e in entries):
                continue
            best = (addr, entries, winner)
            break

        if best is None:
            raise RuntimeError("routing_longest_prefix: could not construct instance after bounded attempts")

        addr, entries, winner = best
        winner_entry = None
        for e in entries:
            if e["hop"] == winner and _matches(bits, e["prefix"], e["plen"], addr):
                winner_entry = e
                break
        assert winner_entry is not None

        table = []
        for e in entries:
            table.append({
                "prefix": _fmt_prefix(bits, e["prefix"], e["plen"]),
                "plen": int(e["plen"]),
                "hop": e["hop"],
                "metric": int(e["metric"]),
            })

        payload = {
            "bits": int(bits),
            "destination": _dst(addr, bits),
            "table": table,
        }
        metadata = edict({"bits": bits, "destination": payload["destination"],
                          "table": table, "answer": winner, "payload": payload})
        return Entry(metadata=metadata, answer=winner)

    def render_prompt(self, metadata):
        lines = []
        lines.append("A router picks the next hop for a destination address from its routing table.")
        lines.append("Every destination and prefix below is a {}-bit binary string.".format(metadata.bits))
        lines.append("")
        lines.append("Routing table (prefix, length -> next hop (metric)):")
        for e in metadata.table:
            lines.append("  {}, /{} -> {} (metric {})".format(e["prefix"], e["plen"], e["hop"], e["metric"]))
        lines.append("")
        lines.append("Destination to route: {}".format(metadata.destination))
        lines.append("")
        lines.append("Selection rules, applied in order:")
        lines.append("  1. An entry matches if its /P prefix equals the destination's first P bits.")
        lines.append("  2. Among matching entries, keep the longest prefix (largest P).")
        lines.append("  3. Ties on prefix length are broken by the lowest metric.")
        lines.append("  4. Remaining ties on length and metric are broken by the lexicographically smallest next hop.")
        lines.append("")
        lines.append("The answer is the selected next hop, written exactly as a hop name like R3 from the table.")
        return "\n".join(lines)

    def distractor_candidates(self, entry):
        addr = int(entry.metadata.destination, 2)
        bits = int(entry.metadata.bits)
        table = [{
            "prefix": int(e["prefix"], 2) << (bits - e["plen"]),
            "plen": int(e["plen"]),
            "hop": e["hop"],
            "metric": int(e["metric"]),
        } for e in entry.metadata.table]
        matching = [e for e in table if _matches(bits, e["prefix"], e["plen"], addr)]
        correct = entry.answer
        out = set()
        if matching:
            max_plen = max(e["plen"] for e in matching)
            longest = [e for e in matching if e["plen"] == max_plen]
            if longest:
                min_metric = min(e["metric"] for e in longest)
                out.add(min(e["hop"] for e in longest if e["metric"] == min_metric))
                out.add(max(longest, key=lambda e: e["metric"])["hop"])
            out.add(max(matching, key=lambda e: e["metric"])["hop"])
        out.discard(correct)
        return list(out)

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        try:
            submitted = str(answer).strip()
        except Exception:
            return 0.0
        return 1.0 if submitted == entry.answer else 0.0
