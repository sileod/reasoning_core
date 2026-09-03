import random

from reasoning_core.template import Task, Entry, Config, edict
from reasoning_core.utils import score_scalar


TASK_META = {'parent_source_id': None,
 'idea': 'Add closed-form reasoning over linear recurrences.',
 'hypothesis': 'S34',
 'changes': 'Ask for a distant term of a recurrence stated in prose with its '
            'initial conditions.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1172182934,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class LinearRecurrenceConfig(Config):
    order: int = 2
    coeff_max: int = 5
    init_max: int = 5
    index: int = 20

    def apply_difficulty(self, level):
        self.index = 10 + 6 * level
        self.coeff_max = 4 + level
        self.init_max = 4 + level
        if level >= 3:
            self.order = 3


def _mat_mul(a, b):
    n = len(a)
    m = len(b[0])
    p = len(b)
    c = [[0] * m for _ in range(n)]
    for i in range(n):
        row = a[i]
        ci = c[i]
        for k in range(p):
            aik = row[k]
            if aik:
                bval = b[k]
                for j in range(m):
                    ci[j] += aik * bval[j]
    return c


def _mat_pow(base, exp):
    n = len(base)
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        res[i][i] = 1
    while exp:
        if exp & 1:
            res = _mat_mul(res, base)
        base = _mat_mul(base, base)
        exp >>= 1
    return res


def _recurrence_term(coeffs, inits, index):
    order = len(coeffs)
    if index < order:
        return inits[index]
    if order == 1:
        base = [[coeffs[0]]]
    elif order == 2:
        base = [[coeffs[0], coeffs[1]], [1, 0]]
    else:
        base = [[coeffs[0], coeffs[1], coeffs[2]], [1, 0, 0], [0, 1, 0]]
    pw = _mat_pow(base, index - order + 1)
    result = 0
    for i in range(order):
        result += pw[0][i] * inits[order - 1 - i]
    return result


def _ones_word(n):
    ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
            'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
            'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    if n < 20:
        return ones[n]
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
            'eighty', 'ninety']
    t, o = divmod(n, 10)
    if o == 0:
        return tens[t]
    return tens[t] + '-' + ones[o]


def _coeff_phrase(c):
    if c == 2:
        return 'twice the previous value'
    if c == -2:
        return 'negative twice the previous value'
    if c == 1:
        return 'the previous value'
    if c == -1:
        return 'the negative of the previous value'
    return str(c) + ' times the previous value'


def _render(payload):
    coeffs = payload['coeffs']
    inits = payload['inits']
    index = payload['index']
    order = payload['order']
    if order == 2:
        inits_text = (f"its first two values are {_ones_word(inits[0])} and "
                      f"{_ones_word(inits[1])}")
        rec_text = (f"{_coeff_phrase(coeffs[0])} plus {abs(coeffs[1])} times "
                    f"the value before that")
        if coeffs[1] < 0:
            rec_text = (f"{_coeff_phrase(coeffs[0])} minus {abs(coeffs[1])} "
                        f"times the value before that")
    else:
        inits_text = (f"its first three values are {_ones_word(inits[0])}, "
                      f"{_ones_word(inits[1])} and {_ones_word(inits[2])}")
        rec_text = (f"{_coeff_phrase(coeffs[0])} plus {abs(coeffs[1])} times "
                    f"the value before that plus {abs(coeffs[2])} times the "
                    f"value two before that")
        if coeffs[1] < 0:
            rec_text = rec_text.replace(' plus ', ' minus ', 1)
        if coeffs[2] < 0:
            rec_text = rec_text.replace(' plus ', ' minus ', 1)
    lines = [
        "A sequence is defined as follows: " + inits_text +
        "; each subsequent value is " + rec_text + ".",
        "What is the value of the " + _ones_word(index) +
        "-th term? (The first value is the 0-th term.)",
        "",
        "Answer with a single integer.",
    ]
    return '\n'.join(lines)


def _generate(config):
    order = config.order
    while True:
        coeffs = [random.randint(-config.coeff_max, config.coeff_max)
                  for _ in range(order)]
        if any(c == 0 for c in coeffs):
            continue
        if order == 2 and coeffs[0] + coeffs[1] in (0, 1, -1, 2, -2):
            continue
        inits = [random.randint(-config.init_max, config.init_max)
                 for _ in range(order)]
        index = config.index
        answer = _recurrence_term(coeffs, inits, index)
        if answer in set(coeffs) or answer in set(inits):
            continue
        return {
            'coeffs': coeffs,
            'inits': inits,
            'index': index,
            'order': order,
            'answer': answer,
        }


class LinearRecurrence(Task):
    config_cls = LinearRecurrenceConfig

    def generate_entry(self):
        data = _generate(self.config)
        metadata = edict({
            'coeffs': data['coeffs'],
            'inits': data['inits'],
            'index': data['index'],
            'order': data['order'],
        })
        metadata.payload = {
            'coeffs': data['coeffs'],
            'inits': data['inits'],
            'index': data['index'],
            'order': data['order'],
        }
        return Entry(metadata=metadata, answer=str(data['answer']))

    def render_prompt(self, metadata):
        return _render(metadata.payload)

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)
