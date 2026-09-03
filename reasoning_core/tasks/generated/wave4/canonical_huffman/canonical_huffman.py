import heapq
import random
from dataclasses import dataclass, field

from reasoning_core.template import Config, Entry, Task, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'Add canonical Huffman coding over a stated symbol frequency table.',
 'hypothesis': 'S40',
 'changes': 'Ask for the codeword of a named symbol under the canonical code, '
            'or the total encoded length of a message.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1963225530,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def canonical_codes(symbols, freqs):
    heap = []
    n = len(symbols)
    for i, sym in enumerate(symbols):
        heapq.heappush(heap, (freqs[i], i, i))
    children = {}
    while len(heap) > 1:
        f1, i1, n1 = heapq.heappop(heap)
        f2, i2, n2 = heapq.heappop(heap)
        new = n + len(children)
        children[new] = (n1, n2)
        heapq.heappush(heap, (f1 + f2, min(i1, i2), new))
    root = heapq.heappop(heap)
    root_node = root[2]
    lengths = {}
    _dfs(root_node, 0, children, lengths, symbols)
    return _canonicalize(symbols, lengths)


def _dfs(node, depth, children, lengths, symbols):
    if node in children:
        left, right = children[node]
        _dfs(left, depth + 1, children, lengths, symbols)
        _dfs(right, depth + 1, children, lengths, symbols)
    else:
        lengths[symbols[node]] = depth


def _canonicalize(symbols, depths):
    order = sorted(symbols, key=lambda s: (depths[s], s))
    code = {}
    num = 0
    prev_len = 0
    for sym in order:
        ln = depths[sym]
        num = num << (ln - prev_len)
        code[sym] = format(num, '0{}b'.format(ln))
        num += 1
        prev_len = ln
    return code


@dataclass
class CanonicalHuffmanConfig(Config):
    n_symbols: int = 6
    max_freq: int = 20
    spread: int = 10

    def apply_difficulty(self, level):
        self.n_symbols = 5 + level
        self.max_freq = 12 + 3 * level
        self.spread = 5 + level


class CanonicalHuffman(Task):
    config_cls = CanonicalHuffmanConfig

    def generate_entry(self):
        cfg = self.config
        while True:
            n = random.randint(cfg.n_symbols, cfg.n_symbols)
            symbols = [chr(ord('a') + i) for i in range(n)]
            base = random.randint(1, cfg.spread)
            freqs = []
            for _ in range(n):
                f = max(1, base + random.randint(0, cfg.max_freq))
                freqs.append(f)
            codes = canonical_codes(symbols, freqs)
            distinct = len(set(len(c) for c in codes.values()))
            if distinct < 2:
                continue
            answer_sym = random.choice(symbols)
            message = ''.join(random.choice(symbols) for _ in range(random.randint(2, 8)))
            total_bits = sum(len(codes[s]) for s in message)
            payload = edict({
                'symbols': list(symbols),
                'freqs': list(freqs),
                'table': {s: f for s, f in zip(symbols, freqs)},
                'target': answer_sym,
                'message': message,
                'total': total_bits,
            })
            metadata = edict({
                'symbols': list(symbols),
                'freqs': [int(f) for f in freqs],
                'target': answer_sym,
                'codeword': codes[answer_sym],
                'message': message,
                'total_bits': total_bits,
            })
            metadata.payload = payload
            mode = random.choice(['codeword', 'length'])
            if mode == 'codeword':
                answer = codes[answer_sym]
            else:
                answer = str(total_bits)
            metadata.mode = mode
            if answer:
                return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        table = metadata.payload['table']
        lines = ['Symbol : Frequency'] + ['{} : {}'.format(s, f) for s, f in table.items()]
        if metadata.mode == 'codeword':
            q = 'What is the canonical Huffman codeword of symbol "{}"?'.format(metadata.target)
            hint = 'The answer is a bit string, for example 0110.'
        else:
            q = ('How many bits in total does the message "{}" occupy under the '
                 'canonical Huffman code?'.format(metadata.message))
            hint = 'The answer is a non-negative integer.'
        return ('A canonical Huffman code is built from a symbol frequency table as '
                'follows. First obtain the code lengths using the usual greedy merge, '
                'breaking ties by the smaller symbol. Then assign codewords in order of '
                'increasing length and, within a length, in alphabetical symbol order.\n'
                + '\n'.join(lines) + '\n\n' + q + '\n\n' + hint)

    def score_answer(self, answer, entry):
        return score_canonical(answer, entry)


def _parse_answer(answer):
    return answer.strip()


class CanonicalHuffmanScorer(object):
    pass


def score_canonical(answer, entry):
    a = _parse_answer(answer)
    gt = entry.answer
    if a == gt:
        return 1.0
    return 0.0
