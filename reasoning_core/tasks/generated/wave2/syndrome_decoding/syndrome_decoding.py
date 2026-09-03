from dataclasses import dataclass
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class SyndromeDecodingConfig(Config):
    n_bits: int = 7
    n_checks: int = 3

    def apply_difficulty(self, level):
        self.n_bits = sround(self.n_bits + 2 * level)
        self.n_checks = sround(self.n_checks + level)


def _int_to_vec(v, m):
    return [(v >> sh) & 1 for sh in range(m - 1, -1, -1)]


def _make_codeword(A):
    m = len(A)
    n = len(A[0])
    M = [row[:] for row in A]
    pivot_row = {}
    free = set(range(n))
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, m):
            if M[r][col]:
                piv = r
                break
        if piv is None:
            continue
        M[row], M[piv] = M[piv], M[row]
        for r in range(m):
            if r != row and M[r][col]:
                M[r] = [x ^ y for x, y in zip(M[r], M[row])]
        pivot_row[col] = row
        free.discard(col)
        row += 1
    free = list(free)
    c = [0] * n
    for f in free:
        if random.getrandbits(1):
            c[f] = 1
    for p, r in pivot_row.items():
        val = 0
        for f in free:
            if M[r][f] and c[f]:
                val ^= 1
        c[p] = val
    return c


class SyndromeDecoding(Task):
    config_cls = SyndromeDecodingConfig

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_bits)
        m = int(cfg.n_checks)
        if n > (1 << m) - 1:
            n = (1 << m) - 1
        allv = [_int_to_vec(v, m) for v in range(1, 1 << m)]
        cols = random.sample(allv, n)
        A = [[cols[j][r] for j in range(n)] for r in range(m)]
        c = _make_codeword(A)
        opts = [None] + list(range(n))
        er = random.choice(opts)
        r = list(c)
        if er is not None:
            r[er] ^= 1
        checks = [[j + 1 for j in range(n) if A[ri][j]] for ri in range(m)]
        corrected = "".join(str(b) for b in c)
        received = "".join(str(b) for b in r)
        payload = {
            "query": "A binary linear code over GF(2) uses 1-based positions. "
                     "It is defined by these parity-check rules: for each rule, "
                     "the XOR of the bits at the listed positions must equal 0.",
            "rules": ["positions %s" % p for p in checks],
            "received_word": received,
            "instruction": "Apply each rule, locate the single corrupted bit if "
                           "there is one (at most one bit is wrong), and give the "
                           "corrected codeword.",
        }
        metadata = edict({
            "n": n,
            "m": m,
            "er": er,
            "received": received,
            "corrected": corrected,
            "payload": payload,
        })
        return Entry(metadata=metadata, answer=corrected)

    def render_prompt(self, metadata):
        return (f"{render_payload(metadata.payload)}\n\n"
                f"The answer is the corrected codeword as a binary string of "
                f"{metadata.n} bits (0s and 1s, no spaces).")

    def score_answer(self, answer, entry):
        try:
            a = str(answer).strip()
        except (TypeError, ValueError):
            return 0.0
        if len(a) != len(entry.answer) or not all(ch in "01" for ch in a):
            return 0.0
        return 1.0 if a == entry.answer else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'Add syndrome decoding over a stated linear code.',
 'hypothesis': 'S20',
 'changes': 'Ask which position was corrupted, or what the corrected word is.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3017200263,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
