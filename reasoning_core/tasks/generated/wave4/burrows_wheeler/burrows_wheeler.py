import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TERMINATOR = "$"


TASK_META = {'parent_source_id': None,
 'idea': 'Add the Burrows-Wheeler transform and its inverse over short '
         'strings.',
 'hypothesis': 'S41',
 'changes': 'Ask for the transform of a string, or for the string a given '
            'transform came from.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 422980222,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class BurrowsWheelerConfig(Config):
    min_len: int = 3
    max_len: int = 6
    alphabet_size: int = 3
    n_letters: int = 5

    def apply_difficulty(self, level):
        self.max_len = sround(self.max_len + level)
        self.alphabet_size = sround(self.alphabet_size + max(0, level - 2))
        self.n_letters = sround(self.n_letters + level)


def _bwt_forward(s):
    rotations = sorted(s[i:] + s[:i] for i in range(len(s)))
    return "".join(r[-1] for r in rotations)


def _bwt_inverse(t):
    n = len(t)
    col = sorted(range(n), key=lambda i: t[i])
    r = col[0]
    out = [""] * n
    for j in range(n):
        r = col[r]
        out[j] = t[r]
    return "".join(out)


def _is_valid_forward(orig):
    bw = _bwt_forward(orig)
    return _bwt_inverse(bw) == orig


def _is_valid_inverse(bw):
    return TERMINATOR in bw


class BurrowsWheeler(Task):
    config_cls = BurrowsWheelerConfig

    def generate_entry(self):
        cfg = self.config
        max_len = cfg.max_len
        alphabet = [chr(ord("a") + i) for i in range(cfg.alphabet_size)]
        n_letters = cfg.n_letters
        term = TERMINATOR

        for _ in range(50):
            length = random.randint(cfg.min_len, max_len)
            orig = "".join(random.choice(alphabet) for _ in range(length)) + term
            bw = _bwt_forward(orig)
            if bw != orig and _bwt_inverse(bw) == orig:
                break
        else:
            raise RuntimeError("failed to build a valid forward instance")

        mode = random.choice(["forward", "inverse"])

        if mode == "forward":
            answer = bw
            prompt_meta = {
                "string": orig[:-1],
                "ask": ("Calculate the Burrows-Wheeler transform of the string. Append the "
                        "terminator character '$' (smallest symbol) to the end of the string, "
                        "take all its rotations, sort them lexicographically, and read the "
                        "last character of each rotation in sorted order. Report the full "
                        "length-%d transform string (which contains exactly one '$')." % (length + 1)),
            }
        else:
            answer = orig[:-1]
            prompt_meta = {
                "string": bw,
                "ask": ("This is the Burrows-Wheeler transform of some original string, "
                        "produced by appending the terminator character '$' (smallest symbol) "
                        "to that string, sorting all its rotations, and taking the last "
                        "character of each rotation in sorted order. Invert the transform "
                        "(using the LF-mapping over the sorted first column) to recover the "
                        "original string and report it as the length-%d string of letters "
                        "(without the '$' terminator)." % length),
            }

        metadata = edict({
            "payload": {"string": prompt_meta["string"], "alphabet": "".join(alphabet)},
            "ask": prompt_meta["ask"],
            "mode": mode,
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return (f"{payload}\n\n{metadata.ask}\n\nThe answer is a string "
                f"whose characters come from the alphabet plus '$'.")


    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip()
        return 1.0 if a == entry.answer else 0.0
