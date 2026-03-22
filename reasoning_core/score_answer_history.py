import hashlib
import inspect
import subprocess
import textwrap
from pathlib import Path


_SCORE_ANSWER_CACHE = {}
_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root():
    return _REPO_ROOT


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def score_answer_hash(task_cls, fn=None):
    fn = fn or task_cls.score_answer
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        source = repr(fn)
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def _score_answer_sources(task_cls):
    return {str(version): spec for version, spec in dict(getattr(task_cls, "score_answer_history", {}) or {}).items()}


def _default_history_path(task_cls):
    source_path = inspect.getsourcefile(task_cls)
    if source_path is None:
        raise RuntimeError(f"Could not locate source file for {task_cls.__name__}")
    source_path = Path(source_path).resolve()
    try:
        return source_path.relative_to(repo_root()).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Task source {source_path} is outside the repository root; set an explicit history path."
        ) from exc


def _resolve_history_callable(task_cls, namespace, spec):
    spec = dict(spec or {})
    callable_name = spec.get("callable")
    if callable_name:
        return namespace[callable_name]

    class_name = spec.get("class_name", task_cls.__name__)
    method_name = spec.get("method", "score_answer")
    history_cls = namespace.get(class_name)
    if history_cls is not None and hasattr(history_cls, method_name):
        return getattr(history_cls, method_name)
    if method_name in namespace:
        return namespace[method_name]
    raise KeyError(f"Could not resolve a historical scorer for {task_cls.task_name} from spec {spec}")


def _load_history_source(task_cls, version, spec):
    spec = {"commit": spec} if isinstance(spec, str) else dict(spec or {})
    if "file" in spec:
        source_path = Path(spec["file"])
        if not source_path.is_absolute():
            source_path = repo_root() / source_path
        return source_path.read_text(), str(source_path), spec

    commit = spec.get("commit")
    if not commit:
        raise KeyError(f"score_answer_history[{version!r}] must define either 'file' or 'commit'.")
    history_path = spec.get("path") or _default_history_path(task_cls)
    source = subprocess.check_output(
        ["git", "show", f"{commit}:{history_path}"],
        cwd=repo_root(),
        text=True,
    )
    return source, f"{commit}:{history_path}", spec


def _load_historical_score_answer(task_cls, version, spec):
    cache_key = (task_cls.__module__, task_cls.__name__, str(version), repr(spec))
    if cache_key in _SCORE_ANSWER_CACHE:
        return _SCORE_ANSWER_CACHE[cache_key]

    source, filename, spec = _load_history_source(task_cls, version, spec)
    namespace = {"__builtins__": __builtins__}
    exec(compile(source, filename, "exec"), namespace)
    fn = _resolve_history_callable(task_cls, namespace, spec)
    _SCORE_ANSWER_CACHE[cache_key] = fn
    return fn


def resolve_score_answer_fn(task_cls, entry=None):
    metadata = getattr(entry, "metadata", None) or (entry.get("metadata", {}) if entry is not None else {}) or {}
    score_meta = metadata.get("_score_answer") or {}
    requested_version = metadata.get("_score_answer_version") or metadata.get("score_answer_version") or score_meta.get("version")
    requested_hash = metadata.get("_score_answer_hash") or score_meta.get("hash")

    current_version = str(getattr(task_cls, "score_answer_version", 0))
    current_hash = score_answer_hash(task_cls, task_cls.score_answer)
    if requested_version is None:
        if requested_hash and requested_hash != current_hash:
            for version, spec in _score_answer_sources(task_cls).items():
                fn = _load_historical_score_answer(task_cls, version, spec)
                if score_answer_hash(task_cls, fn) == requested_hash:
                    return fn
        return task_cls.score_answer

    requested_version = str(requested_version)
    if requested_version == current_version:
        return task_cls.score_answer

    spec = _score_answer_sources(task_cls).get(requested_version)
    if spec is None:
        if requested_hash == current_hash:
            return task_cls.score_answer
        raise KeyError(
            f"Unknown score_answer version {requested_version!r} for task {task_cls.task_name}. "
            f"Set score_answer_history[{requested_version!r}] to a legacy file or commit."
        )
    return _load_historical_score_answer(task_cls, requested_version, spec)
