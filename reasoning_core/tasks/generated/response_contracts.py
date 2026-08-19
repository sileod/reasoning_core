import json
import random
import string
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround


def _json(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _json_score(answer, entry):
    try:
        predicted = json.loads(str(answer).strip())
        reference = json.loads(entry["answer"] if isinstance(entry, dict) else entry.answer)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    return float(predicted == reference)


@dataclass
class SchemaBoundQueryConfig(Config):
    n_rows: int = 6
    n_groups: int = 3
    magnitude: int = 9
    max_attempts: int = 100

    def apply_difficulty(self, level):
        self.n_rows = sround(self.n_rows + 1.3 * level)
        self.n_groups = sround(self.n_groups + 0.25 * level)
        self.magnitude = sround(self.magnitude + 1.1 * level)
        self.max_attempts = sround(self.max_attempts + 10 * level)


class SchemaBoundQuery(Task):
    summary = "Execute a record query while satisfying a sampled exact nested JSON schema."
    config_cls = SchemaBoundQueryConfig

    def generate_entry(self):
        cfg = self.config
        groups = list(string.ascii_uppercase[:max(2, min(cfg.n_groups, 8))])
        for _ in range(cfg.max_attempts):
            rows = [
                {"id": f"R{i+1}", "group": random.choice(groups), "value": random.randint(-cfg.magnitude, cfg.magnitude)}
                for i in range(cfg.n_rows)
            ]
            group = random.choice(groups)
            threshold = random.randint(-cfg.magnitude, cfg.magnitude)
            selected = [r for r in rows if r["group"] == group and r["value"] >= threshold]
            if not selected or len(selected) == len(rows):
                continue
            schema = random.choice(("flat", "nested", "records"))
            ids = [r["id"] for r in selected]
            total = sum(r["value"] for r in selected)
            if schema == "flat":
                answer = {"ids": ids, "count": len(ids), "total": total}
            elif schema == "nested":
                answer = {"selection": {"ids": ids, "count": len(ids)}, "aggregate": {"total": total}}
            else:
                answer = {"matches": [{"id": r["id"], "value": r["value"]} for r in selected], "total": total}
            return Entry(metadata=edict(rows=rows, group=group, threshold=threshold, schema=schema), answer=_json(answer))
        raise RuntimeError("Failed to generate a nontrivial schema-bound query")

    def render_prompt(self, m):
        rows = "\n".join(f"{r['id']}: group={r['group']}, value={r['value']}" for r in m.rows)
        schemas = {
            "flat": '{"ids":[string,...],"count":integer,"total":integer}',
            "nested": '{"selection":{"ids":[string,...],"count":integer},"aggregate":{"total":integer}}',
            "records": '{"matches":[{"id":string,"value":integer},...],"total":integer}',
        }
        return (
            f"Records:\n{rows}\n\nSelect records with group={m.group} and value >= {m.threshold}, preserving input order.\n"
            f"Answer as JSON matching exactly this schema, with no extra keys or prose:\n{schemas[m.schema]}"
        )

    def score_answer(self, answer, entry):
        return _json_score(answer, entry)

    def balancing_key(self, problem):
        return problem.metadata.schema


@dataclass
class ConditionalResponseContractConfig(Config):
    n_records: int = 6
    n_rules: int = 2
    magnitude: int = 12
    max_attempts: int = 100

    def apply_difficulty(self, level):
        self.n_records = sround(self.n_records + 0.9 * level)
        self.n_rules = min(4, sround(self.n_rules + 0.35 * level))
        self.magnitude = sround(self.magnitude + 1.4 * level)
        self.max_attempts = sround(self.max_attempts + 10 * level)


def _condition(rule, winner):
    kind, arg, _, _ = rule
    if kind == "even":
        return winner["score"] % 2 == 0
    if kind == "high":
        return winner["score"] >= arg
    if kind == "group":
        return winner["group"] == arg
    if kind == "flag":
        return winner["flag"] == "yes"
    raise ValueError(kind)


def _apply_rule(text, rule):
    _, _, action, token = rule
    if action == "append":
        return f"{text} {token}"
    if action == "prepend":
        return f"{token} {text}"
    if action == "wrap":
        left, right = token
        return f"{left}{text}{right}"
    raise ValueError(action)


def _rule_text(rule):
    kind, arg, action, token = rule
    cond = {
        "even": "the winner's score is even",
        "high": f"the winner's score is at least {arg}",
        "group": f"the winner's group is {arg}",
        "flag": "the winner's flag is yes",
    }[kind]
    effect = {
        "append": f"append token {token}",
        "prepend": f"prepend token {token}",
        "wrap": f"wrap the entire current answer in {token[0]} and {token[1]}",
    }[action]
    return f"If {cond}, {effect}; otherwise do nothing."


class ConditionalResponseContract(Task):
    summary = "Solve a selection problem and execute output transformations whose activation depends on the semantic result."
    config_cls = ConditionalResponseContractConfig

    def generate_entry(self):
        cfg = self.config
        groups = ("A", "B", "C")
        actions = (("append", "EVEN"), ("prepend", "GROUP"), ("append", "HIGH"), ("wrap", ("[", "]")))
        for _ in range(cfg.max_attempts):
            rows = [
                {"id": f"R{i+1}", "score": random.randint(0, cfg.magnitude), "eligible": random.choice(("yes", "yes", "no")),
                 "group": random.choice(groups), "flag": random.choice(("yes", "no"))}
                for i in range(cfg.n_records)
            ]
            eligible = [r for r in rows if r["eligible"] == "yes"]
            if not eligible:
                continue
            winner = min(eligible, key=lambda r: (-r["score"], r["id"]))
            cutoff = random.randint(1, cfg.magnitude)
            catalog = [
                ("even", None, *actions[0]),
                ("group", random.choice(groups), *actions[1]),
                ("high", cutoff, *actions[2]),
                ("flag", None, *actions[3]),
            ]
            rules = random.sample(catalog, min(cfg.n_rules, len(catalog)))
            states = [_condition(r, winner) for r in rules]
            if len(rules) > 1 and (all(states) or not any(states)):
                continue
            answer = winner["id"]
            for active, rule in zip(states, rules):
                if active:
                    answer = _apply_rule(answer, rule)
            return Entry(metadata=edict(rows=rows, rules=rules, winner=winner["id"], states=states), answer=answer)
        raise RuntimeError("Failed to generate a mixed conditional-contract instance")

    def render_prompt(self, m):
        rows = "\n".join(
            f"{r['id']}: score={r['score']}, eligible={r['eligible']}, group={r['group']}, flag={r['flag']}" for r in m.rows
        )
        rules = "\n".join(f"{i}. {_rule_text(r)}" for i, r in enumerate(m.rules, 1))
        return (
            f"Records:\n{rows}\n\nChoose the eligible record with the largest score; break ties by lexicographically smallest ID. "
            "Start the answer as that ID. Then apply these rules in order to the current answer:\n"
            f"{rules}\nThe answer is the final transformed string and nothing else."
        )

    def score_answer(self, answer, entry):
        reference = entry["answer"] if isinstance(entry, dict) else entry.answer
        return float(str(answer).strip() == str(reference).strip())

    def balancing_key(self, problem):
        return tuple(problem.metadata.states)


@dataclass
class ProtectedSpanTransformationConfig(Config):
    n_items: int = 6
    magnitude: int = 8
    max_attempts: int = 100

    def apply_difficulty(self, level):
        self.n_items = sround(self.n_items + 1.1 * level)
        self.magnitude = sround(self.magnitude + 1.3 * level)
        self.max_attempts = sround(self.max_attempts + 10 * level)


def _protected_token(i):
    a, b = random.sample(string.ascii_letters, 2)
    return f"<{a}{i:02d}:{b}-{random.randint(0, 9)}>"


class ProtectedSpanTransformation(Task):
    summary = "Select and transform records while preserving opaque protected spans byte-for-byte."
    config_cls = ProtectedSpanTransformationConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            items = [{"token": _protected_token(i), "value": random.randint(-cfg.magnitude, cfg.magnitude)} for i in range(cfg.n_items)]
            a = random.choice((-3, -2, 2, 3))
            b = random.randint(-cfg.magnitude, cfg.magnitude)
            parity = random.choice((0, 1))
            selected = [dict(x, result=a * x["value"] + b) for x in items if abs(x["value"]) % 2 == parity]
            if not selected or len(selected) == len(items):
                continue
            selected.sort(key=lambda x: (x["result"], x["token"]))
            answer = "\n".join(f"{x['token']}={x['result']}" for x in selected)
            return Entry(metadata=edict(records=items, a=a, b=b, parity=parity), answer=answer)
        raise RuntimeError("Failed to generate a nontrivial protected-span transformation")

    def render_prompt(self, m):
        items = "\n".join(f"{x['token']} value={x['value']}" for x in m.records)
        parity = "even" if m.parity == 0 else "odd"
        sign = "+" if m.b >= 0 else "-"
        return (
            f"Items:\n{items}\n\nKeep exactly the items whose absolute original value is {parity}. "
            f"For each kept item compute {m.a}*value {sign} {abs(m.b)}. Sort kept items by the computed value ascending, then by protected span. "
            "Each answer line is PROTECTED_SPAN=COMPUTED_VALUE. Copy every protected span exactly, including case and punctuation."
        )

    def score_answer(self, answer, entry):
        reference = entry["answer"] if isinstance(entry, dict) else entry.answer
        norm = lambda x: "\n".join(line.strip() for line in str(x).strip().splitlines())
        return float(norm(answer) == norm(reference))

    def balancing_key(self, problem):
        return "even" if problem.metadata.parity == 0 else "odd"
