import os
import sys
import shutil
import subprocess
import time
import uuid
import re
import contextlib
import signal
import fcntl
import appdirs

# ============================================================================
# CONFIGURATION & DEBUGGING
# ============================================================================

DEBUG = True  # Set to True to see timing logs in console

APP_NAME = "prover_tools"
BASE_DIR = appdirs.user_cache_dir(APP_NAME)
SIF_DIR = os.path.join(BASE_DIR, "images")
BUILD_TMP_DIR = os.path.join(BASE_DIR, "tmp_build")

UDOCKER_CMD_TIMEOUT = 120

# Detect Backend
APPTAINER_BIN = shutil.which('apptainer') or shutil.which('singularity')
USE_APPTAINER = bool(APPTAINER_BIN)

def log(msg):
    if DEBUG:
        print(f"[ProverTools {os.getpid()}] {msg}", file=sys.stderr)

# ============================================================================
# UTILS
# ============================================================================

def _run_subprocess_safe(args, timeout=30, **kwargs):
    old_handler = signal.signal(signal.SIGALRM, signal.SIG_IGN)
    remaining = signal.alarm(0)
    try:
        return subprocess.run(args, timeout=timeout, **kwargs)
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        if remaining > 0: signal.alarm(remaining)
        elif remaining == 0 and old_handler not in (signal.SIG_IGN, signal.SIG_DFL, None):
            signal.alarm(1)

@contextlib.contextmanager
def _file_lock(lock_path):
    """Simple file lock to prevent race conditions when copying to RAM."""
    with open(lock_path, 'w') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def _get_udocker_dir():
    if 'UDOCKER_DIR' in os.environ: return os.environ['UDOCKER_DIR']
    candidates = [os.path.expanduser('~'), '/dev/shm', os.getenv('TMPDIR'), '/tmp']
    user = os.getenv('USER', 'user')
    for base in filter(None, candidates):
        if os.access(base, os.W_OK) and not (os.statvfs(base).f_flag & os.ST_NOEXEC):
            return os.path.join(base, f'udocker-env-{user}')
    raise RuntimeError("No suitable directory found for UDOCKER_DIR")

@contextlib.contextmanager
def _build_env_context():
    """Point Apptainer to NFS only during build to save local space."""
    if not USE_APPTAINER:
        yield
        return

    os.makedirs(BUILD_TMP_DIR, exist_ok=True)
    os.makedirs(SIF_DIR, exist_ok=True)

    old_env = dict(os.environ)
    vars_to_set = ['APPTAINER_TMPDIR', 'APPTAINER_CACHEDIR', 'SINGULARITY_TMPDIR', 'SINGULARITY_CACHEDIR']
    
    for v in vars_to_set: os.environ[v] = BUILD_TMP_DIR
    try:
        yield
    finally:
        for v in vars_to_set:
            if v in old_env: os.environ[v] = old_env[v]
            else: del os.environ[v]

def _ensure_apptainer_image_persistence(docker_image):
    """
    Step 1: Ensure image exists on NFS (Slow Build).
    Returns path to the SIF on NFS.
    """
    sanitized = re.sub(r'[\W]+', '_', docker_image).strip('_')
    sif_path_nfs = os.path.join(SIF_DIR, f"{sanitized}.sif")
    
    if os.path.exists(sif_path_nfs):
        return sif_path_nfs

    with _build_env_context():
        log(f"Building Apptainer image on NFS: {sif_path_nfs} ...")
        subprocess.run(
            [APPTAINER_BIN, "build", sif_path_nfs, f"docker://{docker_image}"], 
            check=True, timeout=1800, stdout=sys.stdout, stderr=sys.stderr
        )
    return sif_path_nfs

def _load_sif_to_ram(nfs_path):
    """
    Step 2: Copy NFS image to /dev/shm (RAM) for speed.
    Uses locking so only one process does the copy.
    """
    fname = os.path.basename(nfs_path)
    ram_path = os.path.join("/dev/shm", fname)
    lock_path = ram_path + ".lock"

    # If already exists and size matches, use it
    if os.path.exists(ram_path):
        if os.path.getsize(ram_path) == os.path.getsize(nfs_path):
            return ram_path

    # Copy with lock
    log(f"Waiting for lock to copy {fname} to RAM...")
    with _file_lock(lock_path):
        # Double check inside lock
        if os.path.exists(ram_path) and os.path.getsize(ram_path) == os.path.getsize(nfs_path):
            return ram_path
        
        log(f"Copying SIF to RAM ({ram_path})... this speeds up execution.")
        t0 = time.time()
        # Copy to temp file first then atomic rename
        temp_ram = ram_path + f".tmp.{os.getpid()}"
        shutil.copy2(nfs_path, temp_ram)
        os.rename(temp_ram, ram_path)
        log(f"Copy finished in {time.time()-t0:.2f}s")
        
    return ram_path

# ============================================================================
# MAIN PROCESS CLASS
# ============================================================================

class Embeded_process:
    def __init__(self, docker_image="valentinq76/tools:2.0", provers_to_check=['vampire', 'eprover']):
        self.docker_image = docker_image
        self.provers = provers_to_check
        self.is_setup = False
        self._lock = contextlib.nullcontext()
        self.sif_path = None
        self.pid = os.getpid()
        
        self.uid = f"{self.pid}-{uuid.uuid4().hex[:8]}"
        self.container_name = f"prover-sess-{self.uid}"
        self.tmpfs_host = os.path.join("/dev/shm", self.container_name)
        # Use /tmp inside container since --contain isolates /dev/shm
        self.tmpfs_cont = f"/tmp/{self.container_name}"

        self.native_paths = {p: shutil.which(p) for p in provers_to_check if shutil.which(p)}

    def setup(self):
        with self._lock:
            if self.is_setup: return
            if os.getpid() != self.pid: raise RuntimeError("Process forked.")

            os.makedirs(self.tmpfs_host, exist_ok=True)

            if USE_APPTAINER:
                # 1. Get/Build on NFS
                nfs_sif = _ensure_apptainer_image_persistence(self.docker_image)
                # 2. Cache in RAM
                self.sif_path = _load_sif_to_ram(nfs_sif)
            else:
                # Fallback to UDocker logic
                os.environ['UDOCKER_DIR'] = _get_udocker_dir()
                os.makedirs(os.environ['UDOCKER_DIR'], exist_ok=True)
                res = subprocess.run(["udocker", "--allow-root", "images"], 
                                   capture_output=True, text=True, timeout=UDOCKER_CMD_TIMEOUT)
                if self.docker_image not in res.stdout:
                    subprocess.run(["udocker", "--allow-root", "pull", self.docker_image], timeout=600)
                subprocess.run(['udocker', "--allow-root", 'rm', self.container_name], capture_output=True)
                subprocess.run(['udocker', "--allow-root", 'create', f'--name={self.container_name}', self.docker_image], capture_output=True)

            self.is_setup = True

    def run_prover(self, solver, options, tptp_file, timeout=30):
        # NATIVE SHORTCUT
        if solver in self.native_paths:
            return _run_subprocess_safe([self.native_paths[solver]] + options + [tptp_file], 
                                      capture_output=True, text=True, timeout=timeout)

        t_start = time.time()
        if not self.is_setup: self.setup()
        
        fname = os.path.basename(tptp_file)
        host_f = os.path.join(self.tmpfs_host, fname)
        cont_f = os.path.join(self.tmpfs_cont, fname)
        shutil.copy(tptp_file, host_f)

        try:
            if USE_APPTAINER:
                # Use --no-home and --contain to reduce startup I/O
                cmd = [APPTAINER_BIN, "exec", "--no-home", "--contain", "--bind", f"{self.tmpfs_host}:{self.tmpfs_cont}", 
                       self.sif_path, solver] + options + [cont_f]
            else:
                cmd = ["udocker", "--allow-root", "run", f"--volume={self.tmpfs_host}:{self.tmpfs_cont}", 
                       self.container_name, solver] + options + [cont_f]

            res = _run_subprocess_safe(cmd, capture_output=True, text=True, timeout=timeout)
            
            # Debugging slow executions
            dur = time.time() - t_start
            if dur > 5.0 and DEBUG:
                log(f"Slow execution ({dur:.2f}s): {' '.join(cmd)}")
                
            return res

        except Exception as e:
            self.kill()
            if "timeout" in str(e).lower(): raise TimeoutError
            raise e
        finally:
            if os.path.exists(host_f): os.remove(host_f)

    def run_agint(self, input_string, timeout=30):
        if not self.is_setup: self.setup()
        
        if USE_APPTAINER:
            cmd = [APPTAINER_BIN, "exec", "--no-home", "--contain", self.sif_path, "AGInTRater", "-c"]
        else:
            cmd = ["udocker", "--allow-root", "run", self.container_name, "AGInTRater", "-c"]

        return _run_subprocess_safe(cmd, input=input_string, capture_output=True, text=True, timeout=timeout).stdout

    def kill(self):
        with self._lock:
            if not self.is_setup: return
            shutil.rmtree(self.tmpfs_host, ignore_errors=True)
            if not USE_APPTAINER:
                subprocess.run(['udocker', '--allow-root', 'rm', '-f', self.container_name], 
                             capture_output=True, timeout=5)
            self.is_setup = False

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.kill()
    def __del__(self): self.kill()

# ============================================================================
# SESSION
# ============================================================================

_prover_session = None

def get_prover_session(docker_image="valentinq76/tools:2.0"):
    global _prover_session
    current_pid = os.getpid()
    
    if _prover_session is None or _prover_session.pid != current_pid:
        _prover_session = Embeded_process(docker_image=docker_image)
    
    return _prover_session

def initialize_prover_session(docker_image="valentinq76/tools:2.0"):
    sess = get_prover_session(docker_image)
    sess.setup()
    return sess

ensure_image = lambda img: None 
prover_session = None