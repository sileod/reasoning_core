import os
import tempfile
from pathlib import Path


HOME = Path.home().resolve()
RC_HOME = Path(os.environ.get("RC_HOME", HOME / ".reasoning_core")).expanduser().resolve()


def scratch_root():
    """Node-local scratch, when the caller declares one.

    Confining every runtime path to HOME forces caches onto whatever HOME points at. On a
    shared filesystem that breaks the HF datasets lock (Errno 116) and kills the job with no
    log, so a node-local root has to be expressible without moving HOME.
    """
    scratch = os.environ.get("RC_SCRATCH")
    return Path(scratch).expanduser().resolve() if scratch else None


def home_path(path, *, name="path"):
    path = Path(path).expanduser().resolve()
    roots = [root for root in (HOME, scratch_root()) if root is not None]
    if not any(path == root or root in path.parents for root in roots):
        allowed = " or ".join(str(root) for root in roots)
        raise ValueError(f"{name} must be inside {allowed}, got {path}")
    return path


home_path(RC_HOME, name="RC_HOME")
RUNS_HOME = RC_HOME / "runs"
CACHE_HOME = RC_HOME / "cache"
TMP_HOME = RC_HOME / "tmp"
LOCKS_HOME = RC_HOME / "locks"


def env_path(name, default):
    return home_path(os.environ.get(name, default), name=name)


def configure_runtime_env():
    scratch = scratch_root()
    tmp = env_path("RC_TMP", scratch / "tmp" if scratch else TMP_HOME)
    hf = env_path("HF_CACHE", scratch / "huggingface" if scratch else CACHE_HOME / "huggingface")
    for path in (tmp, hf):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "HF_HOME": str(hf),
        "HF_DATASETS_CACHE": str(hf / "datasets"),
        "TORCHINDUCTOR_CACHE_DIR": str(tmp / "torchinductor"),
        "TRITON_CACHE_DIR": str(tmp / "triton"),
        "WANDB_DIR": str(RUNS_HOME / "wandb"),
        "WANDB_CACHE_DIR": str(CACHE_HOME / "wandb"),
        "TMPDIR": str(tmp),
        "TEMP": str(tmp),
        "TMP": str(tmp),
        "TOKENIZERS_PARALLELISM": "false",
    })
    tempfile.tempdir = str(tmp)
