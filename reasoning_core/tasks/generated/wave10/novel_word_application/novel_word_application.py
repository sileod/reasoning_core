"""Use newly defined words productively in unfamiliar grammatical constructions and combinations.

Each entry defines a handful of novel stems (invented syllables plus a semantic gloss),
then poses a target meaning and asks the reader to realize it in the given novel
mini-language --- mixing a compound construction (head/modifier ordering with a stated
linking rule) and a derivational construction (an affix that turns a noun into an agent
or an abstract quality). The answer is always the joined novel form, so it never equals
a word that appears on the surface and it varies across every example.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


_SYLLABLES = [
    "ba", "be", "bi", "bo", "bu", "da", "de", "di", "do", "du",
    "ga", "ge", "gi", "go", "gu", "ka", "ke", "ki", "ko", "ku",
    "la", "le", "li", "lo", "lu", "ma", "me", "mi", "mo", "mu",
    "na", "ne", "ni", "no", "nu", "pa", "pe", "pi", "po", "pu",
    "ra", "re", "ri", "ro", "ru", "sa", "se", "si", "so", "su",
    "ta", "te", "ti", "to", "tu", "za", "ze", "zi", "zo", "zu",
]

_GLOSSES = [
    "bird", "mountain river", "woven basket", "hidden door",
    "storm cloud", "round stone", "night flower", "metal hook",
    "low bench", "dry leaf", "fishing net", "clay jug",
    "wooden flute", "small lantern", "gravel path", "thatched roof",
]

_LINKERS = ["", "o", "i"]

_AGENT_SUFFIXES = ["-am", "-ek", "-or"]
_QUALITY_SUFFIXES = ["-ush", "-il", "-at"]


def _make_stem(rng):
    while True:
        stem = rng.choice(_SYLLABLES) + rng.choice(_SYLLABLES)
        if stem not in _RESERVED:
            _RESERVED.add(stem)
            return stem


_RESERVED = set()


def _build_compound(rng, cfg):
    """Mode A: a head noun plus one or two modifiers joined with a randomized
    head/modifier order and a stated linking rule. The answer is the unique
    concatenation that the stated construction produces."""
    n_mods = rng.choice([1, 2]) if cfg.level >= 2 else 1

    modifier_before = rng.random() < 0.5
    linker = rng.choice(_LINKERS)

    head = (_make_stem(rng), rng.choice(_GLOSSES))
    used = {head[1]}
    mods = []
    for _ in range(n_mods):
        g = rng.choice(_GLOSSES)
        while g in used:
            g = rng.choice(_GLOSSES)
        used.add(g)
        mods.append((_make_stem(rng), g))

    if modifier_before:
        entry_order = mods + [head]
    else:
        entry_order = [head] + mods

    parts = []
    for idx, (stem, _) in enumerate(entry_order):
        link = linker if (idx > 0 and linker and stem[:1] in "bgdklmnprstz") else ""
        parts.append(link + stem)
    answer = "".join(parts)

    defines = [f"{stem} = a {gloss}" for stem, gloss in entry_order]

    target = f"a {head[1]}"
    if n_mods == 1:
        target += f" that is a {mods[0][1]}"
    else:
        target += " that is " + " and ".join(f"a {g}" for _, g in mods)

    position = "before" if modifier_before else "after"
    if linker:
        link_txt = (
            f"the vowel '{linker}' is inserted between the parts whenever the part "
            f"that follows starts with a consonant"
        )
    else:
        link_txt = "the parts are joined directly with no linking vowel"

    prompt_body = (
        f"In an invented language the following new words are defined, in the order "
        f"they will be combined:\n"
        + "\n".join(f"  {d}" for d in defines) + "\n\n"
        f"To make a compound for something in this language, every modifier is written "
        f"{position} the head noun; {link_txt}. The parts are combined in the order "
        f"listed above."
    )

    return {"stems": [s for s, _ in entry_order],
            "glosses": dict(entry_order),
            "defines": defines, "position": position, "linker": linker,
            "answer": answer, "head": head, "mods": mods,
            "target": target}, answer, target, prompt_body


def _build_derivation(rng, cfg):
    """Mode B: an affix changes a noun's grammatical role (agent or abstract quality)."""
    if rng.random() < 0.5:
        suffix = rng.choice(_AGENT_SUFFIXES)
        gloss = rng.choice(_GLOSSES)
        stem = _make_stem(rng)
        affixed = stem + suffix.lstrip("-")
        target = f"one who works with or handles a {gloss}"
        order = "appended to the end of the noun"
    else:
        suffix = rng.choice(_QUALITY_SUFFIXES)
        gloss = rng.choice(_GLOSSES)
        stem = _make_stem(rng)
        affixed = stem + suffix.lstrip("-")
        target = f"the quality or abstract idea of being a {gloss}"
        order = "appended to the end of the noun"

    prompt_body = (
        f"In an invented language the new word is defined:\n"
        f"  {stem} = a {gloss}\n\n"
        f"To derive a related word meaning \"{target}\", the suffix '{suffix}' is "
        f"{order}."
    )
    return {"stem": stem, "gloss": gloss, "suffix": suffix, "answer": affixed,
            "target": target}, affixed, target, prompt_body


@dataclass
class NovelWordApplicationConfig(Config):
    level: int = 0

    def apply_difficulty(self, level):
        self.level = level


class NovelWordApplication(Task):
    summary = "Apply newly defined words in compounding (head/modifier order, linking vowel, multi-modifier) and derivational (agent and abstract-quality affixes) constructions in an invented language."
    config_cls = NovelWordApplicationConfig

    def generate_entry(self):
        rng = random
        if rng.random() < 0.5:
            info, answer, target, prompt_body = _build_compound(rng, self.config)
        else:
            info, answer, target, prompt_body = _build_derivation(rng, self.config)

        gold = answer
        assert isinstance(gold, str) and gold

        metadata = edict({
            "info": info,
            "target": target,
            "prompt_body": prompt_body,
            "answer": answer,
        })
        metadata.payload = {"body": prompt_body, "target": target}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"{metadata.prompt_body}\n\n"
            f"Write, in this invented language, the form that expresses:\n"
            f"  {metadata.target}\n\n"
            f"The answer is that single novel word, in lowercase, with no spaces."
        )

    def score_answer(self, answer, entry):
        gold = entry.answer
        if not isinstance(answer, str):
            return 0.0
        a = "".join(ch for ch in answer.lower() if ch.isalpha())
        return 1.0 if a == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'novel_word_application (draw 1 of 2)',
 'hypothesis': 'ASTRA0-13',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/novel_word_application',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1621874809,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
