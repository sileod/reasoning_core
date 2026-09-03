import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


class KmpPrefixValueConfig(Config):
    n: int = 8
    alphabet: int = 3

    def apply_difficulty(self, level):
        self.n = sround(self.n + 2 * level)
        self.alphabet = sround(self.alphabet + (level > 2))


def _prefix_function(pattern):
    pi = [0] * len(pattern)
    for i in range(1, len(pattern)):
        j = pi[i - 1]
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j
    return pi


class KmpPrefixValue(Task):
    summary = ("Given a pattern and a position, output the KMP prefix-function value at "
               "that position, over variable-length patterns on small alphabets at "
               "selected positions; diagonal values are excluded and other positions "
               "may be hit.")
    config_cls = KmpPrefixValueConfig

    def generate_entry(self):
        level = self.config.level
        n_base = self.config.n
        alphabet = self.config.alphabet

        while True:
            length = max(2, level + random.randrange(max(1, n_base - level), n_base + 1))
            pattern = [random.randrange(alphabet) for _ in range(length)]
            pos = random.randrange(length)
            if pos == 0:
                continue
            pi = _prefix_function(pattern)
            value = pi[pos]
            if value < 0 or value >= pos:
                continue
            break

        text = "".join(chr(97 + c) for c in pattern)
        metadata = edict({
            "pattern": text,
            "position": pos,
            "pi": pi,
        })
        metadata.payload = {"pattern": text, "position": pos}
        answer = str(value)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            "The KMP prefix function of a string is defined as: prefix[i] is the length "
            "of the longest proper prefix of the string that is also a suffix of the "
            "substring ending at index i. Compute this value at the given position for "
            "the given pattern.\n"
            f"{render_payload(metadata.payload)}\n\n"
            "The answer is the prefix-function value, an integer."
        )

    def score_answer(self, answer, entry):
        try:
            val = int(float(str(answer).strip()))
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if val == int(entry.answer) else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'kmp_prefix_value (draw 1 of 2)',
 'hypothesis': 'W1-072',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/kmp_prefix_value',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 161918849,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
