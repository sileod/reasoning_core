import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload
from reasoning_core.utils import score_scalar

TASK_META = {'parent_source_id': None,
 'idea': 'Add benefit or premium determination under rules with exceptions.',
 'hypothesis': 'S21',
 'changes': 'Ask for the amount a stated policy yields for a described case, '
            'not for an eligibility label.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3335976966,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class PolicyConfig(Config):
    n_rules: int = 3
    n_exceptions: int = 1
    n_distractors: int = 1
    value_span: int = 60

    def apply_difficulty(self, level):
        self.n_rules = 3 + level
        self.n_exceptions = 1 + (level >= 2) + (level >= 4)
        self.n_distractors = 1 + level
        self.value_span = 60 + 40 * level


def _fmt(x):
    return f"{x:,}"


def build_policy(config):
    n_rules = config.n_rules
    n_exc = config.n_exceptions
    span = config.value_span

    while True:
        base = random.randint(10, 40) * 100
        rates = [random.randint(5, 20) / 100 for _ in range(n_rules)]
        caps = [random.randint(1, 5) * 1000 for _ in range(n_rules)]
        thresholds = [random.randint(2, 8) * 1000 for _ in range(n_rules)]

        exc_kind = []
        for _ in range(n_exc):
            exc_kind.append(random.choice(['flat', 'waive', 'half']))

        def compute(case_amount, exc_used):
            remaining = case_amount
            total = 0
            for i in range(n_rules):
                rate = rates[i]
                cap = caps[i]
                if i < n_exc and exc_used[i]:
                    if exc_kind[i] == 'flat':
                        rate = 0.0
                    elif exc_kind[i] == 'waive':
                        cap = 0
                    else:
                        rate = rate / 2
                contrib = min(remaining * rate, cap)
                total += contrib
                remaining -= contrib
            return total

        case = random.randint(500, span * 100)
        exc_used = [random.random() < 0.5 for _ in range(n_exc)]
        exc_trigger = [
            random.randint(500, span * 100) for _ in range(n_exc)
        ]
        use_exc = [bool(exc_used[i] and case >= exc_trigger[i]) for i in range(n_exc)]

        total = compute(case, use_exc)
        total = int(round(total))

        if total > 0 and all(r >= 0.05 for r in rates):
            distractors = []
            while len(distractors) < config.n_distractors:
                d = random.randint(200, span * 100)
                if d != case and d not in distractors:
                    distractors.append(d)
            return {
                'base': base,
                'rates': rates,
                'caps': caps,
                'thresholds': thresholds,
                'case': case,
                'exc_used': exc_used,
                'exc_trigger': exc_trigger,
                'exc_kind': exc_kind,
                'distractors': distractors,
                'total': total,
                'n_rules': n_rules,
                'n_exc': n_exc,
            }


def render_rules(policy):
    lines = []
    base = policy['base']
    lines.append(
        f"The base benefit is {_fmt(base)} per claim before any adjustments. "
        "The adjustment tiers apply in the order listed below; each tier's amount "
        "is subtracted from the remaining claim balance before the next tier applies."
    )
    for i in range(policy['n_rules']):
        rate = policy['rates'][i]
        cap = policy['caps'][i]
        pct = int(round(rate * 100))
        suffix = ""
        if i < policy['n_exc']:
            kind = policy['exc_kind'][i]
            trig = policy['exc_trigger'][i]
            cond = "at least" if policy['exc_used'][i] else "at most"
            if kind == 'flat':
                exc = "no adjustment is applied"
            elif kind == 'waive':
                exc = "the cap is waived"
            else:
                exc = "the adjustment is halved"
            suffix = f" Exception: for a claim balance {cond} {_fmt(trig)}, {exc}."
        lines.append(
            f"Tier {i+1}: {pct}% of the remaining balance, up to a cap of "
            f"{_fmt(cap)}{suffix}"
        )
    return lines


def total_amount(policy):
    return policy['total']


class PolicyDetermination(Task):
    config_cls = PolicyConfig

    def generate_entry(self):
        cfg = self.config
        policy = build_policy(cfg)
        rules_text = "\n".join(render_rules(policy))

        distractor_lines = []
        if policy['distractors']:
            vals = [f"{_fmt(d)}" for d in policy['distractors']]
            distractor_lines.append(
                "The claim log also records these figures: " + ", ".join(vals) + "."
            )

        case_text = f"The case under review has a claim balance of {_fmt(policy['case'])}."

        excuse_text = ""
        for i in range(policy['n_exc']):
            if policy['exc_used'][i] and policy['case'] >= policy['exc_trigger'][i]:
                kind = policy['exc_kind'][i]
                if kind == 'flat':
                    note = "the tier-{} exception applies (flat)".format(i + 1)
                elif kind == 'waive':
                    note = "the tier-{} exception applies (waive)".format(i + 1)
                else:
                    note = "the tier-{} exception applies (half)".format(i + 1)
                excuse_text = note
        if excuse_text:
            case_text += " " + excuse_text.capitalize() + "."

        payload = {
            "policy": rules_text,
            "case": case_text,
        }
        if distractor_lines:
            payload["record"] = distractor_lines[0]

        answer = str(policy['total'])

        metadata = edict({
            "policy": policy,
            "payload": payload,
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            "Determine the benefit amount after all listed adjustments are applied "
            "in the stated order. The answer is an integer number of currency units; "
            "give only that number."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)
