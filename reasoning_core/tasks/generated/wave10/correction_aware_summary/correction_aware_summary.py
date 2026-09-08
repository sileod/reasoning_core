import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'correction_aware_summary (draw 1 of 2)',
 'hypothesis': 'ASTRA0-14',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/correction_aware_summary',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3123569846,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

_DETAILS = ["uses", "picks", "buys", "sells", "wants", "keeps", "drafts", "reserves"]


@dataclass
class CorrectionAwareSummaryConfig(Config):
    n_offer: int = 2
    n_correction: int = 1
    wording: int = 3

    def apply_difficulty(self, level):
        self.n_offer = sround(self.n_offer + level)
        self.n_correction = sround(1 + level // 2)
        self.wording = sround(self.wording + level)


class CorrectionAwareSummary(Task):
    summary = ("Read a short exchange of commitments with follow-up corrections and "
               "withdrawals, and report the final set of commitments in a stable "
               "alphabetical listing, excluding superseded and rejected details.")
    config_cls = CorrectionAwareSummaryConfig

    def generate_entry(self):
        c = self.config
        n_offer = int(c.n_offer)
        n_correct = int(c.n_correction)

        while True:
            topics = random.sample(_DETAILS, n_offer)
            roles = ["A", "B", "C", "D", "E"][:max(2, n_offer)]
            planner = random.choice(roles)
            items = [f"{random.choice(['x', 'y', 'z', 'w', 'q', 'r'])}{i+1}" for i in range(n_offer)]
            verb = random.choice(["X", "Y", "Z"])
            object_word = random.choice(["box", "bundle", "set", "lot", "pack"])
            obj = f"{random.choice(['red', 'blue', 'green', 'dark', 'large', 'small'])} {object_word}"

            commits = list(zip(topics, items))

            start = f"{roles[0]} {random.choice(['uses', 'picks', 'takes', 'orders'])} the {obj}."
            exchanges = []
            for (topic, item) in commits:
                exchanges.append(f"{planner} {topic} {item}.")
            sentence = start + " " + " ".join(exchanges)

            final = list(commits)

            for i in range(n_correct):
                t = random.randint(0, len(final) - 1)
                topic, item = final[t]
                kind = random.random()
                if kind < 0.45:
                    new_item = f"{random.choice(['x', 'y', 'z', 'w', 'q', 'r'])}{random.randint(1, 9)}"
                    corrected = (topic, new_item)
                    sentence += f" Later, {planner} corrects {topic}: it is {new_item}, not {item}."
                    final[t] = corrected
                elif kind < 0.8:
                    sentence += f" Afterwards, {planner} drops the plan to {topic} {item}."
                    final.pop(t)
                    if len(final) == 0:
                        break
                else:
                    sentence += f" Then {planner} reaffirms {topic} {item}."
            if len(final) == 0:
                continue
            final = sorted(final)
            answer = "; ".join(f"{t} {i}" for t, i in final)
            self._sentence = sentence
            metadata = edict({
                "exchange": sentence,
                "final": final,
                "answer": answer,
            })
            metadata.payload = {"exchange": sentence}
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (f"{render_payload(metadata.payload)}\n\n"
                f"Summarize the final commitments the group actually holds at the end of the "
                f"exchange. Exclude anything later corrected or withdrawn. Name the answer as a "
                f"list of \"topic item\" pairs, sorted alphabetically by topic then item, "
                f"joined by \"; \" with no surrounding brackets.")

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        gold = entry.answer
        norm = " ".join(str(answer).split()).lower()
        gold_norm = " ".join(gold.split()).lower()
        if norm == gold_norm:
            return 1.0
        return 0.0
