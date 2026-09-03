import random
from dataclasses import dataclass
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'Add exact series-parallel reduction over resistor networks.',
 'hypothesis': 'S30',
 'changes': 'Ask for the equivalent resistance of a described network as an '
            'exact fraction.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 564967272,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def leaf():
    val = random.randint(1, 12)
    return ("resistor", Fraction(val))


def build(depth_remaining):
    if depth_remaining == 0 or random.random() < 0.35:
        return leaf()
    kind = random.choice(["series", "parallel"])
    left = build(depth_remaining - 1)
    right = build(depth_remaining - 1)
    return (kind, left, right)


def describe(node):
    if node[0] == "resistor":
        return f"a {node[1].numerator} ohm resistor"
    kind, left, right = node
    left_txt = describe(left)
    right_txt = describe(right)
    if kind == "series":
        return f"{left_txt} connected in series with {right_txt}"
    return f"{left_txt} connected in parallel with {right_txt}"


def evaluate(node):
    if node[0] == "resistor":
        return node[1]
    kind, left, right = node
    a = evaluate(left)
    b = evaluate(right)
    if kind == "series":
        return a + b
    return (a * b) / (a + b)


def total_resistors(node):
    if node[0] == "resistor":
        return 1
    return total_resistors(node[1]) + total_resistors(node[2])


def valid_tree(node):
    r = evaluate(node)
    if r <= 0 or r.numerator <= 0 or r.denominator <= 0:
        return False
    total = Fraction(sum(x[1] for x in leaves(node)))
    if r > total:
        return False
    return True


def leaves(node):
    if node[0] == "resistor":
        yield node
    else:
        yield from leaves(node[1])
        yield from leaves(node[2])


@dataclass
class ResistorNetworkConfig(Config):
    depth: int = 3
    min_resistors: int = 3
    max_resistors: int = 6

    def apply_difficulty(self, level):
        self.depth = 2 + level
        self.min_resistors = 2 + level
        self.max_resistors = 4 + 2 * level


class ResistorNetworks(Task):
    config_cls = ResistorNetworkConfig

    def generate_entry(self):
        depth = self.config.depth
        min_r = self.config.min_resistors
        max_r = self.config.max_resistors
        node = None
        while node is None:
            candidate = build(depth)
            n = total_resistors(candidate)
            if min_r <= n <= max_r and valid_tree(candidate):
                node = candidate
        answer = evaluate(node)
        assert answer > 0, "resistance must be strictly positive"
        total = sum(x[1] for x in leaves(node))
        assert answer <= total, "resistance cannot exceed sum of all resistors"
        description = describe(node)
        metadata = edict({
            "network": description,
            "n_resistors": total_resistors(node),
        })
        metadata.payload = {"network": description}
        return Entry(
            metadata=metadata,
            answer=f"{answer.numerator}/{answer.denominator}",
        )

    def render_prompt(self, metadata):
        return (
            f"A circuit contains {metadata.payload['network']}.\n\n"
            "What is the equivalent resistance of the whole circuit? "
            "Give the answer as an exact fraction in lowest terms, "
            "e.g. 7/3."
        )

    def score_answer(self, answer, entry):
        text = str(answer).strip()
        if "/" in text:
            try:
                num, den = text.split("/")
                return 1.0 if Fraction(int(num.strip()), int(den.strip())) == Fraction(entry.answer) else 0.0
            except (ValueError, ZeroDivisionError):
                return 0.0
        try:
            cand = Fraction(text)
        except (ValueError, ZeroDivisionError):
            return 0.0
        return 1.0 if cand == Fraction(entry.answer) else 0.0
