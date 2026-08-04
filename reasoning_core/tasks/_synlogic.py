"""Synlogic wrapper: MiniMax-AI/SynLogic games as one reasoning_core Task.

Games are DISCOVERED agnostically from the SynLogic checkout (no hardcoded task/class list): every
games/tasks/<t> whose module exposes a `games.base.game.Game` subclass is registered, and the verifier
comes from SynLogic's own task2verifier map. Only games that emit a concrete gold `answer` are usable for
answer-only training (SynLogic also has verifier-only grid puzzles with empty answers) — `usable_games()`
returns that subset. Metadata follows the source_task / source_collection convention (bare subtask name).

Set SYNLOGIC_ROOT to use a checkout; otherwise an importable install is used, or the repo is cloned under
appdirs. Deps: numpy nltk sympy math-verify pandas.
"""
import json
import random
import inspect
import logging
import importlib
import importlib.util
import os
import re
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from appdirs import user_cache_dir
from reasoning_core.template import Task, Entry, Config

SYNLOGIC_REPO = "https://github.com/MiniMax-AI/SynLogic.git"

_REGISTRY = None   # {task_name: Game subclass}
_USABLE = None     # [task_name, ...] with a concrete gold answer


_INIT_LANGUAGE = {
    "object_counting": {"en": "en", "zh": "zh"},
    "operation": {"en": "english", "zh": "chinese"},
    "word_sorting_mistake": {"en": "english", "zh": "chinese"},
}
_GENERATE_LANGUAGE = {
    "boolean_expressions": {"en": "en", "zh": "zh"},
    "space_reasoning": {"en": "en", "zh": "cn"},
    "space_reasoning_tree": {"en": "en", "zh": "cn"},
}
_DIFFICULTY_MAX = {
    "sudoku": 4,
    "web_of_lies": 5,
}


@contextmanager
def _quiet_synlogic():
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        yield


def _norm_language(language):
    language = str(language).lower()
    if language in {"en", "eng", "english"}:
        return "en"
    if language in {"zh", "cn", "chinese"}:
        return "zh"
    return "mixed"


def _has_cjk(text):
    return bool(re.search(r"[\u3400-\u9fff]", str(text)))


def _has_importable_synlogic():
    return importlib.util.find_spec("games") and importlib.util.find_spec("task2verifier")


@lru_cache(maxsize=1)
def _synlogic_root():
    env_root = os.environ.get("SYNLOGIC_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        if not (root / "games" / "tasks").is_dir():
            raise RuntimeError(f"SYNLOGIC_ROOT does not look like a SynLogic checkout: {root}")
        return root

    if _has_importable_synlogic():
        return None

    if os.environ.get("SYNLOGIC_AUTO_DOWNLOAD", "1").lower() in {"0", "false", "no"}:
        raise RuntimeError("Synlogic requires SynLogic importable, SYNLOGIC_ROOT set, or auto-download enabled.")

    root = Path(user_cache_dir("reasoning_core")) / "SynLogic"
    if not (root / "games" / "tasks").is_dir():
        if root.exists():
            raise RuntimeError(f"SynLogic cache exists but is incomplete: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", SYNLOGIC_REPO, str(root)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return root


def _on_path():
    root = _synlogic_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    logging.getLogger("games").setLevel(logging.WARNING)  # SynLogic games are chatty at INFO
    return root


def _tasks_dir():
    root = _on_path()
    if root is not None:
        return root / "games" / "tasks"
    games = importlib.import_module("games")
    return Path(games.__file__).resolve().parent / "tasks"


def discover_games():
    """Agnostic registry: import each games/tasks/<t> module and grab its Game subclass. Cached."""
    global _REGISTRY
    if _REGISTRY is None:
        _on_path()
        from games.base.game import Game
        td = _tasks_dir()
        reg = {}
        for t in sorted(os.listdir(td)):
            if not os.path.isdir(os.path.join(td, t)):
                continue
            for mod_name in (f"games.tasks.{t}.scripts.{t}", f"games.tasks.{t}.scripts.{t}_generator"):
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:
                    continue
                cls = next((o for _, o in inspect.getmembers(mod, inspect.isclass)
                            if issubclass(o, Game) and o is not Game and o.__module__ == mod.__name__), None)
                if cls:
                    reg[t] = cls
                    break
        if not reg:
            raise RuntimeError("No SynLogic games discovered; install SynLogic dependencies or set SYNLOGIC_ROOT.")
        _REGISTRY = reg
    return _REGISTRY


def usable_games():
    """Discovered games that emit a concrete gold answer (trainable answer-only). Cached; test-gens once."""
    global _USABLE
    if _USABLE is None:
        u = []
        for t, cls in discover_games().items():
            try:
                with _quiet_synlogic():
                    d = cls().generate(1)
                if d and str(d[0].answer).strip():
                    u.append(t)
            except Exception:
                pass
        _USABLE = sorted(u)
    return _USABLE


@dataclass
class SynlogicConfig(Config):
    task: str = "mixed"           # A discovered game name, or "mixed" to sample discovered games.
    language: str = "en"          # "en", "zh", or "mixed" where supported by SynLogic.
    difficulty: int = 1

    def apply_difficulty(self, level):
        self.difficulty = min(5, self.difficulty + level)


class Synlogic(Task):
    summary = "Execute reasoning games and tasks integrated from the SynLogic framework."
    task_name = "synlogic"

    def __init__(self, config=None, *args, **kwargs):
        _on_path()
        super().__init__(config or SynlogicConfig(), *args, **kwargs)
        self._reg = discover_games()
        self._games = {}

    def _game(self, name):
        language = _norm_language(self.config.language)
        key = (name, language)
        if key not in self._games:
            kwargs = {}
            if language != "mixed" and name in _INIT_LANGUAGE:
                kwargs["language"] = _INIT_LANGUAGE[name][language]
            self._games[key] = self._reg[name](**kwargs)
        return self._games[key]

    def _generate_kwargs(self, name):
        language = _norm_language(self.config.language)
        kwargs = {}
        if language != "mixed" and name in _GENERATE_LANGUAGE:
            kwargs["language"] = _GENERATE_LANGUAGE[name][language]
        if name in _DIFFICULTY_MAX:
            kwargs["difficulty"] = min(self.config.difficulty, _DIFFICULTY_MAX[name])
        return kwargs

    def _accepts_language(self, name):
        return name in _INIT_LANGUAGE or name in _GENERATE_LANGUAGE

    def _accept_prompt(self, name, question):
        language = _norm_language(self.config.language)
        if language != "en":
            return True
        return not _has_cjk(question)

    def _candidate_names(self):
        t = str(self.config.task)
        if t in self._reg:
            return [t] * 24
        names = list(_USABLE or self._reg)
        random.shuffle(names)
        return names

    def generate_entry(self):
        global _USABLE
        for name in self._candidate_names():
            try:
                with _quiet_synlogic():
                    d = self._game(name).generate(1, **self._generate_kwargs(name))[0]
            except Exception:
                continue
            if str(d.answer).strip() and self._accept_prompt(name, d.question):
                break
        else:
            raise RuntimeError("No SynLogic game generated a concrete answer.")

        if _USABLE is not None and name not in _USABLE:
            _USABLE.append(name)
        max_level = _DIFFICULTY_MAX.get(name, 1) - 1
        effective_level = min(self.config.difficulty - 1, max_level)
        meta = dict(d.metadata or {}) | {
            "task_name": f"synlogic.{name}",
            "source_collection": "synlogic",
            "source_task": name,
            "effective_level": effective_level,
            "max_level": max_level,
            "difficulty": d.difficulty,
            "_question": d.question,
        }
        return Entry(json.loads(json.dumps(meta, default=str)), str(d.answer))

    def render_prompt(self, metadata):
        return metadata._question

    def score_answer(self, answer, entry):
        _on_path()
        answer = str(answer).strip()
        reference = str(entry.get("answer", "")).strip()
        if answer == reference:
            return 1.0
        with _quiet_synlogic():
            from task2verifier import verifier_classes
            from base.data import Data
        md = entry["metadata"]
        name = md.get("source_task") or str(md.get("source_dataset", "")).split(".")[-1]
        try:
            data = Data(question=md.get("_question", ""), answer=str(entry.get("answer", "")),
                        difficulty=md.get("difficulty", 1), metadata=md)
            candidates = (answer, f"\\boxed{{{answer}}}", f"<answer>{answer}</answer>")
            with _quiet_synlogic():
                return float(any(verifier_classes[name]().verify(data, c) for c in candidates))
        except Exception:
            return 0.0
