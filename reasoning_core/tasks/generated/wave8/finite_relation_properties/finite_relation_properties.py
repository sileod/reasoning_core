import random

from reasoning_core.template import Task, Entry, Config, edict


class FiniteRelationPropertiesConfig(Config):
    n: int = 4
    density: float = 0.5
    level: int = 0

    def apply_difficulty(self, level):
        self.level = level
        self.n = 2 + level


class FiniteRelationProperties(Task):
    summary = "Given a finite relation, output which of reflexive, symmetric, antisymmetric, and transitive hold."

    config_cls = FiniteRelationPropertiesConfig

    def generate_entry(self):
        n = self.config.n
        while True:
            rel = set()
            for i in range(n):
                for j in range(n):
                    if random.random() < 0.5:
                        rel.add((i, j))
            props = []
            reflexive = all((i, i) in rel for i in range(n))
            symmetric = all(((j, i) in rel) for (i, j) in rel)
            antisymmetric = all((i == j or (j, i) not in rel) for (i, j) in rel)
            transitive = all(
                ((i, k) in rel) for (i, j) in rel for (j2, k) in rel if j == j2
            )
            if reflexive:
                props.append("reflexive")
            if symmetric:
                props.append("symmetric")
            if antisymmetric:
                props.append("antisymmetric")
            if transitive:
                props.append("transitive")
            if props:
                props = sorted(props)
                break
        answer = ", ".join(props)
        metadata = edict({
            "n": int(n),
            "relation": [[int((i, j) in rel) for j in range(n)] for i in range(n)],
            "answer": answer,
        })
        metadata.payload = {"relation": metadata.relation}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        rows = "\n".join(
            "[{}]".format(" ".join(str(x) for x in row)) for row in metadata.relation
        )
        return (
            "A relation is a set of ordered pairs over the set {0, 1, ..., "
            "%d}. The relation is shown as an adjacency matrix below where a "
            "1 in row i, column j means (i, j) is in the relation.\n\n%s\n\n"
            "State which of reflexive, symmetric, antisymmetric, and "
            "transitive hold. The answer is the comma-separated list of the "
            "properties that hold, each chosen from the words reflexive, "
            "symmetric, antisymmetric, transitive. For example "
            "\"symmetric, transitive\"." % (metadata.n - 1, rows)
        )

    def score_answer(self, answer, entry):
        expected = entry.answer
        norm = "".join(answer.split()).lower()
        en = "".join(expected.split()).lower()
        if norm == en:
            return 1.0
        normwords = ",".join(w.strip() for w in answer.split(",") if w.strip())
        enwords = expected
        if normwords.lower() == enwords.lower():
            return 1.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'finite_relation_properties (draw 1 of 2)',
 'hypothesis': 'W1-024',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/finite_relation_properties',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2261582844,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
