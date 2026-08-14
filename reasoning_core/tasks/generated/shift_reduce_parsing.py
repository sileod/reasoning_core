import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


@dataclass
class ShiftReduceParsingConfig(Config):
    n_rules: int = 5
    derivation_depth: int = 6
    query_fraction: float = 0.65
    max_attempts: int = 300

    def apply_difficulty(self, level):
        self.n_rules = sround(self.n_rules + 0.5 * level)
        self.derivation_depth = sround(self.derivation_depth + 1.0 * level)
        self.query_fraction = min(0.9, self.query_fraction + 0.03 * level)
        self.max_attempts = sround(self.max_attempts + 30 * level)


def _reduce_stack(stack, rules):
    reductions = 0
    while True:
        matches = []
        for idx, (lhs, rhs) in enumerate(rules):
            if len(rhs) <= len(stack) and tuple(stack[-len(rhs):]) == rhs:
                matches.append((len(rhs), -idx, lhs, rhs))
        if not matches:
            return reductions
        _, neg_idx, lhs, rhs = max(matches)
        del stack[-len(rhs):]
        stack.append(lhs)
        reductions += 1


def _parse_prefix(tokens, rules, k):
    stack = []
    reductions = 0
    for token in tokens[:k]:
        stack.append(token)
        reductions += _reduce_stack(stack, rules)
    return stack, reductions


def _generate_reducible_grammar(cfg):
    terminals = list("abcde")
    nonterms = [f"N{i}" for i in range(max(3, cfg.n_rules))]
    rules = []
    for i in range(len(nonterms)):
        if i < 2 or random.random() < 0.45:
            rhs = (random.choice(terminals),)
        else:
            pool = nonterms[:i]
            rhs = tuple(random.choice(pool) for _ in range(random.choice((2, 2, 3))))
        rule = (nonterms[i], rhs)
        if rule not in rules:
            rules.append(rule)
    return terminals, rules


def _expand_symbol(symbol, rules, depth):
    choices = [rhs for lhs, rhs in rules if lhs == symbol]
    if not choices or depth <= 0:
        return [symbol]
    rhs = random.choice(choices)
    out = []
    for x in rhs:
        if any(lhs == x for lhs, _ in rules):
            out.extend(_expand_symbol(x, rules, depth - 1))
        else:
            out.append(x)
    return out


class ShiftReduceParsing(GeneratedMixin, Task):
    summary = "Execute a deterministic shift-reduce parser and report one compact stack state."
    config_cls = ShiftReduceParsingConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            terminals, rules = _generate_reducible_grammar(cfg)
            start = rules[-1][0]
            tokens = _expand_symbol(start, rules, cfg.derivation_depth)
            if not tokens or any(t not in terminals for t in tokens) or len(tokens) < 4:
                continue
            k = min(len(tokens), max(2, round(len(tokens) * cfg.query_fraction)))
            stack, reductions = _parse_prefix(tokens, rules, k)
            if reductions < 2 or len(stack) < 1:
                continue
            answer = " ".join(stack)
            metadata = edict(rules=[(lhs, list(rhs)) for lhs, rhs in rules], tokens=tokens, k=k)
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("Failed to generate shift-reduce instance")

    def render_prompt(self, metadata):
        rules = "\n".join(f"R{i}: {lhs} -> {' '.join(rhs)}" for i, (lhs, rhs) in enumerate(metadata.rules))
        return (
            f"Rules:\n{rules}\nInput: {' '.join(metadata.tokens)}\n"
            "Shift tokens left to right. After every shift, repeatedly reduce the longest stack suffix matching a rule RHS; "
            "ties use the lowest rule number.\n"
            f"What is the stack after consuming {metadata.k} tokens? The answer is the stack symbols from bottom to top, space-separated."
        )

    def score_answer(self, answer, entry):
        norm = lambda x: " ".join(str(x).split())
        return float(norm(answer) == norm(entry.answer))

    def balancing_key(self, problem):
        return len(problem.answer.split())
