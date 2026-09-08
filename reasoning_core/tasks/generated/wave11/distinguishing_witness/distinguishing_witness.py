import random
import re
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'distinguishing_witness (draw 1 of 2)',
 'hypothesis': 'ASTRA2-distinguishing_witness',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave11/distinguishing_witness',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2600584944,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def parse_indices(answer):
    if not isinstance(answer, str):
        return None
    nums = re.findall(r"\d+", answer)
    if not nums:
        return None
    try:
        inds = sorted({int(m) for m in nums})
    except ValueError:
        return None
    return inds


def render_group(indices, weights):
    return ", ".join("item %d" % i for i in sorted(indices))


def smallest_witness(weights, C, D):
    n = len(weights)
    best = None
    best_card = None
    for mask in range(1, 1 << n):
        inds = [i for i in range(n) if mask & (1 << i)]
        total = sum(weights[i] for i in inds)
        fits = total <= C
        heavy = total >= D
        if fits == heavy:
            continue
        tup = tuple(sorted(inds))
        card = len(tup)
        if best is None or card < best_card or (card == best_card and tup < best):
            best = tup
            best_card = card
    return best, best_card


def render_prompt(weights, C, D, gold, level):
    lines = []
    lines.append(
        "A truck has a weight capacity of %d kg. It carries a group of items, "
        "and the group's total weight is the sum of its items' weights." % C)
    wdesc = ", ".join("item %d weighs %d kg" % (i, weights[i]) for i in range(len(weights)))
    lines.append("The available items are: " + wdesc + ".")
    lines.append("")
    lines.append("Consider two statements about a group of items:")
    lines.append("- Statement F: \"The group fits in the truck.\"")
    lines.append("  (true exactly when the group's total weight is at most %d kg)" % C)
    lines.append("- Statement A: \"The group weighs at least %d kg.\"" % D)
    lines.append("  (true exactly when the group's total weight is at least %d kg)" % D)
    lines.append("")
    lines.append(
        "Because %d < %d, the two statements can never both be true at once." % (C, D))
    lines.append("")
    lines.append(
        "Find the smallest nonempty group of items such that the two statements "
        "have DIFFERENT truth values (exactly one of them holds). Here 'smallest' "
        "means fewest items, and if two groups have the same number of items, take "
        "the one whose sorted item indices are lexicographically smallest.")
    lines.append(
        "Give the answer as the item indices of that group, sorted ascending, "
        "e.g. \"item 2, item 3, item 7\".")
    return "\n".join(lines)


@dataclass
class DistinguishingWitnessConfig(Config):
    n_items: int = 5

    def apply_difficulty(self, level):
        self.n_items = 5 + level


class DistinguishingWitness(Task):
    summary = ("Construct the smallest weighted-item group, as a sorted index "
               "set, on which a fit-capacity and a minimum-weight statement have "
               "different truth values.")
    config_cls = DistinguishingWitnessConfig

    def generate_entry(self):
        n = self.config.n_items
        for _ in range(200):
            C = random.randint(12, 30)
            D = C + random.randint(17, max(18, int(5 * C)))
            weights = [random.randint(C + 1, C + 16) for _ in range(n)]
            gold, card = smallest_witness(weights, C, D)
            if gold is None or card < 2:
                continue
            total = sum(weights[i] for i in gold)
            fits = total <= C
            heavy = total >= D
            assert card >= 1
            assert fits != heavy
            assert total <= C or total >= D
            answer = render_group(gold, weights)
            metadata = edict({
                "weights": [int(w) for w in weights],
                "capacity": int(C),
                "demand": int(D),
                "gold_indices": list(gold),
                "level": self.config.level,
            })
            metadata.payload = {
                "weights": metadata.weights,
                "capacity": metadata.capacity,
                "demand": metadata.demand,
            }
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("no distinguishing witness after bounded attempts")

    def render_prompt(self, metadata):
        return render_prompt(
            metadata.weights, metadata.capacity, metadata.demand,
            metadata.gold_indices, metadata.level)

    def score_answer(self, answer, entry):
        parsed = parse_indices(answer)
        if parsed is None:
            return 0.0
        gold = entry.metadata.gold_indices
        if parsed != gold:
            return 0.0
        total = sum(entry.metadata.weights[i] for i in parsed)
        fits = total <= entry.metadata.capacity
        heavy = total >= entry.metadata.demand
        if fits == heavy:
            return 0.0
        return 1.0
