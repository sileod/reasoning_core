import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'diff_patch_application (draw 1 of 1)',
 'hypothesis': 'HV-071',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/diff_patch_application',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 911077872,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _apply_hunks(tokens, hunks):
    tokens = list(tokens)
    for op, start, count, insert in hunks:
        if op == 'ins':
            tokens = tokens[:start] + insert + tokens[start:]
        else:
            tokens = tokens[:start] + insert + tokens[start + count:]
    return tokens


def _runify(tokens):
    return ",".join(tokens)


def _parse_answer(s):
    s = s.strip()
    if s == "":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [t for t in s.split(",") if t != ""]


@dataclass
class DiffPatchConfig(Config):
    n_tokens: int = 6
    n_hunks: int = 2
    vocab_size: int = 5

    def apply_difficulty(self, level):
        self.n_tokens = 6 + level * 2
        self.n_hunks = 2 + level
        self.vocab_size = 5 + level


class DiffPatchApplication(Task):
    summary = "Apply ordered context-aware insertion, deletion, and replacement hunks to text or token sequences, returning the exact resulting content."
    config_cls = DiffPatchConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_tokens
        h = cfg.n_hunks
        vocab = [f"w{i}" for i in range(cfg.vocab_size)]

        for _ in range(300):
            initial = [random.choice(vocab) for _ in range(n)]
            tokens = list(initial)
            hunks = []
            ops = ['ins', 'del', 'rep']
            depth = 0
            while depth < h:
                maxlen = len(tokens)
                op = random.choice(ops)
                if op == 'ins':
                    start = random.randint(0, maxlen)
                    ins_len = random.randint(1, 2)
                    insert = [random.choice(vocab) for _ in range(ins_len)]
                    hunks.append(('ins', start, 0, insert))
                elif op == 'del':
                    count = random.randint(1, min(2, maxlen))
                    start = random.randint(0, maxlen - count)
                    hunks.append(('del', start, count, []))
                else:
                    count = random.randint(1, min(2, maxlen))
                    start = random.randint(0, maxlen - count)
                    ins_len = random.randint(1, 2)
                    insert = [random.choice(vocab) for _ in range(ins_len)]
                    hunks.append(('rep', start, count, insert))
                tokens = _apply_hunks(tokens, [hunks[-1]])
                depth += 1
            if len(tokens) >= 1:
                break
        else:
            raise RuntimeError("no valid instance")

        initial_str = _runify(initial)
        result = _apply_hunks(initial, hunks)
        answer = "[" + _runify(result) + "]"

        hunks_json = [
            {"op": op, "start": int(start), "count": int(count), "insert": list(insert)}
            for op, start, count, insert in hunks
        ]

        metadata = edict({
            "initial": initial_str,
            "hunks": hunks_json,
            "result": answer,
            "n_tokens": int(n),
            "n_hunks": int(h),
            "vocab_size": int(cfg.vocab_size),
        })
        metadata.payload = {"initial": initial_str, "hunks": hunks_json}

        assert "[" + _runify(_apply_hunks(initial_str.split(","), hunks)) + "]" == answer
        assert len(answer) > 2

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        hunks_str = "; ".join(
            f"{h['op']}@{h['start']}x{h['count']}[{','.join(h['insert'])}]"
            for h in metadata.hunks
        )
        return (
            f"Given the token sequence, apply each hunk in order. "
            f"A hunk is 'op@startxcount[insert]' where op is ins (insert the "
            f"bracket tokens before position start), del (delete count tokens "
            f"starting at start), or rep (replace count tokens starting at start "
            f"with the bracket tokens). Positions are relative to the current "
            f"sequence after all preceding hunks have been applied, starting at 0.\n\n"
            f"Initial sequence:\n{metadata.initial}\n\n"
            f"Hunks to apply in order:\n{hunks_str}\n\n"
            f"Give the resulting token sequence as a comma-separated list enclosed "
            f"in brackets and nothing else. Example format: [w0,w2,w1]\n\n"
            f"The answer is the resulting sequence in brackets."
        )

    def score_answer(self, answer, entry):
        gold = _runify(
            _apply_hunks(
                entry.metadata.initial.split(","),
                [
                    (h["op"], int(h["start"]), int(h["count"]), list(h["insert"]))
                    for h in entry.metadata.hunks
                ],
            )
        )
        got = _runify(_parse_answer(answer))
        return 1.0 if got == gold else 0.0
