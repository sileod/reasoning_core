"""SemVer precedence comparison: which of two version strings is higher."""
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

_ALPHANUM = ("alpha", "beta", "rc", "gamma", "dev", "zeta", "omega")


def _parse_ident(text):
    if text.isdigit():
        return ("n", int(text))
    return ("a", text)


def _split_pre(version):
    main, _, pre = version.partition("-")
    major, minor, patch = main.split(".")
    pre_key = None if pre == "" else tuple(_parse_ident(i) for i in pre.split("."))
    return (int(major), int(minor), int(patch), pre_key)


def _version_key(version):
    return _split_pre(version)


def _key_less(a, b):
    for i in range(3):
        if a[i] != b[i]:
            return a[i] < b[i]
    ap, bp = a[3], b[3]
    if ap is None and bp is None:
        return False
    if ap is None:
        return False
    if bp is None:
        return True
    for x, y in zip(ap, bp):
        xk = 0 if x[0] == "n" else 1
        yk = 0 if y[0] == "n" else 1
        if xk != yk:
            return xk < yk
        if x[1] != y[1]:
            return x[1] < y[1]
    return len(ap) < len(bp)


def higher_precedence(a, b):
    ka, kb = _version_key(a), _version_key(b)
    if ka == kb:
        return "equal"
    if _key_less(ka, kb):
        return b
    return a


def _identifier(max_pre_val):
    if random.random() < 0.5:
        return str(random.randint(0, max_pre_val))
    return random.choice(_ALPHANUM)


def _core(max_comp):
    return ".".join(str(random.randint(0, max_comp)) for _ in range(3))


def _make_version(max_comp, max_pre, force_pre):
    v = _core(max_comp)
    if force_pre:
        n_ids = random.randint(1, max_pre)
        ids = [_identifier(max_comp * 3 + 1) for _ in range(n_ids)]
        v += "-" + ".".join(ids)
    return v


@dataclass
class SemVerPrecedenceConfig(Config):
    max_comp: int = 3
    pre_prob: float = 0.2
    max_pre_ids: int = 2
    deep_prob: float = 0.2
    equal_prob: float = 0.08

    def apply_difficulty(self, level):
        self.max_comp = sround(3 + 2 * level)
        self.pre_prob = min(0.15 + 0.13 * level, 0.9)
        self.max_pre_ids = sround(1 + level)
        self.deep_prob = min(0.2 + 0.12 * level, 0.85)
        self.equal_prob = 0.08 - 0.005 * level
        if self.equal_prob < 0.02:
            self.equal_prob = 0.02


class SemanticVersionPrecedence(Task):
    summary = "Given two SemVer strings (core and optional dotted prerelease), determine and name the string with higher precedence by numeric-then-lexical dotted rules, or 'equal'."
    config_cls = SemVerPrecedenceConfig

    def generate_entry(self):
        cfg = self.config
        max_comp = int(cfg.max_comp)
        max_pre = int(cfg.max_pre_ids)
        while True:
            force_a = random.random() < float(cfg.pre_prob)
            force_b = random.random() < float(cfg.pre_prob)
            a = _make_version(max_comp, max_pre, force_a)
            b = _make_version(max_comp, max_pre, force_b)
            if random.random() < float(cfg.deep_prob) and (force_a or force_b):
                core = _core(max_comp)
                pa = _make_version(max_comp, max_pre, True).split("-", 1)[1]
                a = core + "-" + pa
                pb = _make_version(max_comp, max_pre, True).split("-", 1)[1]
                b = core + "-" + pb
            if random.random() < float(cfg.equal_prob):
                b = a
            answer = higher_precedence(a, b)
            if answer != "equal":
                other = b if answer == a else a
                assert higher_precedence(other, answer) == answer
                gold_other = higher_precedence(a, b)
                assert gold_other == answer
            else:
                assert _version_key(a) == _version_key(b)
            payload = {"version a": a, "version b": b}
            metadata = edict({"a": a, "b": b})
            metadata.payload = payload
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"Under Semantic Versioning, precedence is determined by comparing major, "
            f"minor, and patch as numbers, and when those are equal, by pre-release "
            f"identifiers: a version without a pre-release is higher than one with a "
            f"pre-release, and pre-release identifiers compare left to right with "
            f"numeric identifiers ordered numerically (and lower than non-numeric ones) "
            f"while non-numeric identifiers compare by ASCII order, and a longer "
            f"pre-release whose prefix matches is higher.\n\n"
            f"{render_payload(metadata.payload)}\n\n"
            f"Which of the two versions has higher precedence? The answer is that "
            f"version string verbatim, or 'equal' if they have equal precedence."
        )

    def score_answer(self, answer, entry):
        gold = entry["answer"]
        if str(answer).strip() == str(gold).strip():
            return 1.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'semantic_version_precedence (draw 1 of 2)',
 'hypothesis': 'W1-080',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/semantic_version_precedence',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 157335281,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
