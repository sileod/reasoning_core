import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class PrefixCodeConfig(Config):
    n_symbols: int = 4
    max_codeword: int = 8
    n_codewords: int = 4
    max_len: int = 12

    def apply_difficulty(self, level):
        self.n_symbols = sround(self.n_symbols + level)
        self.max_codeword = sround(self.max_codeword + level)
        self.n_codewords = sround(self.n_codewords + level)
        self.max_len = sround(self.max_len + 3 * level)


def _prefix_free(codewords):
    lst = list(codewords)
    n = len(lst)
    for i in range(n):
        for j in range(n):
            if i != j and lst[j].startswith(lst[i]):
                return False
    return True


class PrefixCodeDecode(Task):
    summary = ("Given a prefix-free binary codebook mapping distinct symbols to codewords and a "
               "concatenated bitstream, output the unique decoded symbol sequence; vary codebook size, "
               "codeword lengths, and bitstream composition.")

    config_cls = PrefixCodeConfig

    def generate_entry(self):
        cfg = self.config
        symbols = [f"s{i}" for i in range(cfg.n_symbols)]

        while True:
            selected = random.sample(symbols, k=min(cfg.n_codewords, cfg.n_symbols))
            codewords = {}
            attempts = 0
            while len(codewords) < len(selected) and attempts < 500:
                attempts += 1
                wanted = len(selected) - len(codewords)
                pool = sorted(set(selected) - set(codewords))
                for sym in pool:
                    length = random.randint(1, cfg.max_codeword)
                    cw = "".join(random.choice("01") for _ in range(length))
                    cand = list(codewords.items()) + [(sym, cw)]
                    if _prefix_free([w for _, w in cand]):
                        codewords[sym] = cw
            if len(codewords) == len(selected) and _prefix_free(list(codewords.values())):
                break

        order = sorted(codewords.keys())
        vocab = {codewords[s]: s for s in order}

        seq = [random.choice(order) for _ in range(cfg.max_len)]
        bitstream = "".join(codewords[s] for s in seq)

        symbols_out = sorted({s: code for code, s in vocab.items()}, key=lambda s: s)
        codebook_lines = "\n".join(f"{code} {symbols_out.index(s)}" for code, s in sorted(vocab.items()))

        mapping = {codewords[s]: s for s in order}
        decoded = []
        i = 0
        while i < len(bitstream):
            for code in sorted(mapping, key=len, reverse=True):
                if bitstream.startswith(code, i):
                    decoded.append(mapping[code])
                    i += len(code)
                    break
            else:
                raise RuntimeError("undecodable stream")

        assert decoded == seq

        answer = " ".join(decoded)
        metadata = edict({
            "symbols": symbols_out,
            "codebook": codebook_lines,
            "bitstream": bitstream,
            "fixed": all(len(v) == len(next(iter(codewords.values()))) for v in codewords.values()),
        })
        metadata.payload = {
            "symbols": symbols_out,
            "codebook": codebook_lines,
            "bitstream": bitstream,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return (f"Below is a prefix-free binary codebook and a concatenated bitstream produced by "
                f"encoding a sequence of symbols with that codebook.\n{payload}\n"
                f"Decode the bitstream into the symbol sequence. The answer is a space-separated list of "
                f"symbols in order, for example: s0 s2 s0.")

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        norm = " ".join(answer.split())
        return 1.0 if norm == entry.answer else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'prefix_code_decode (draw 1 of 2)',
 'hypothesis': 'W1-070',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/prefix_code_decode',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1934978647,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
