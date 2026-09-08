import ast
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'presupposition_repair (draw 2 of 2)',
 'hypothesis': 'ASTRA0-15',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/presupposition_repair',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3360584995,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _dist_to_num(fact):
    s = fact.split()
    return int(s[2]) if s[0].isdigit() else None


@dataclass
class PresuppositionRepairV2Config(Config):
    facts: int = 3
    value_range: int = 20

    def apply_difficulty(self, level):
        self.facts = 3 + level
        self.value_range = 20 + 10 * level


class PresuppositionRepair(Task):
    summary = ("Correct a question's false premise using supplied facts, then "
               "answer the conditional; facts give integers, distances and coin "
               "counts; the premise is an equality or inequality over two values "
               "and the answer is either the sum or the sum minus 3.")
    config_cls = PresuppositionRepairV2Config

    def generate_entry(self):
        cfg = self.config
        facts = []
        locations = set()
        values = {}
        for _ in range(cfg.facts):
            loc = "%s%d" % (random.choice("ABCDEFGH"), random.randrange(100))
            while loc in locations:
                loc = "%s%d" % (random.choice("ABCDEFGH"), random.randrange(100))
            locations.add(loc)
            val = random.randrange(2, cfg.value_range + 1)
            values[loc] = val
            kind = random.choice(["int", "dist", "mult"])
            if kind == "dist":
                facts.append("%s is %d meters away" % (loc, val))
            elif kind == "mult":
                facts.append("%s holds %d coins" % (loc, val))
            else:
                facts.append("%s = %d" % (loc, val))

        ls = sorted(locations)
        a, b = ls[0], ls[1]
        va, vb = values[a], values[b]
        total = va + vb

        premise_op = random.choice([">=", "<", "=="])
        if premise_op == "==":
            premise = "%s == %s" % (a, b)
            true = va == vb
        elif premise_op == ">=":
            premise = "%s >= %s" % (a, b)
            true = va >= vb
        else:
            premise = "%s < %s" % (a, b)
            true = va < vb

        answer = total if true else total - 3
        if answer < 0:
            answer = total

        question = "what is the sum of %s and %s?" % (a, b)
        metadata = edict({
            "facts": facts,
            "question": question,
            "premise": premise,
            "premise_true": true,
            "a": a,
            "b": b,
        })
        metadata.payload = {"facts": facts, "question": question}
        return Entry(metadata=metadata, answer=str(answer))

    def render_prompt(self, metadata):
        body = render_payload(metadata.payload)
        return (
            "%s\n\n"
            "The question below carries a premise '%s'. Check that premise "
            "against the facts. If the premise is true, answer the question as "
            "stated. If the premise is false, repair it by subtracting 3 from "
            "the question's answer.\n\nThe answer is an integer."
            % (body, metadata.premise)
        )

    def score_answer(self, answer, entry):
        gold = str(entry.answer)
        try:
            got = str(answer).strip()
        except Exception:
            return 0.0
        try:
            return 1.0 if int(got) == int(gold) else 0.0
        except Exception:
            return 0.0
