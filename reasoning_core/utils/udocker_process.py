import os
import sys
import shutil
import subprocess
import threading
import time
import uuid
import shutil
import os

def get_optimal_udocker_dir():
    """
    Finds the best UDOCKER_DIR by testing standard locations for writability and executability.
    """
    def is_suitable(path):
        try:
            return os.access(path, os.W_OK) and not (os.statvfs(path).f_flag & os.ST_NOEXEC)
        except FileNotFoundError:
            return False

    user = os.getenv('USER', 'user')
    candidates = [
        '/dev/shm',
        os.getenv('TMPDIR'),
        '/tmp',
        os.path.expanduser('~')
    ]

    for base_path in filter(None, candidates):
        if is_suitable(base_path):
            return os.path.join(base_path, f'udocker-env-{user}')
    
    raise RuntimeError("Could not find a suitable directory for UDOCKER_DIR.")

# Set up the environment
os.environ['UDOCKER_DIR'] = get_optimal_udocker_dir()
os.makedirs(os.environ['UDOCKER_DIR'], exist_ok=True)

def manage_stale_containers(dry_run=True):
    """
    Finds stale containers by filesystem mtime and either reports them (dry_run=True)
    or deletes them using 'udocker rm' (dry_run=False).
    """
    try:
        containers_path = os.path.join(os.environ['UDOCKER_DIR'], 'containers')
        cutoff = time.time() - 3600  # 1 hour ago

        for container_id in os.listdir(containers_path):
            path = os.path.join(containers_path, container_id)
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                found_stale = True
                if dry_run:
                    print(f"[WOULD DELETE] Stale container ID: {container_id}")
                else:
                    subprocess.run(["udocker", "--allow-root", "rm", "-f", container_id], capture_output=True)
        

    except (FileNotFoundError, KeyError):
        pass # Silently ignore if UDOCKER_DIR is not set or path is missing


manage_stale_containers(dry_run=False)


import re
import contextlib

def ensure_image(img):
    result = subprocess.run(["udocker", "--allow-root", "images"], capture_output=True, text=True)
    if img not in result.stdout:
        pull_result = subprocess.run(["udocker", "--allow-root", "pull", img])
        if pull_result.returncode != 0:
            raise RuntimeError(f"Failed to pull image {img}")


class Embeded_process:
    """
    A class to manage and interact with a udocker container via subprocess.
    It keeps a container active and automatically stops it after 
    a period of inactivity to save resources.
    """
    def __init__(self, docker_image="valentinq76/tools:2.0", idle_timeout=120,provers_to_check = ['vampire', 'eprover']):
        """
        Args:
            docker_image (str): The Docker image containing the solvers.
            idle_timeout (int): Time in seconds before stopping the inactive container.
        """
        self.docker_image = docker_image
        ensure_image(docker_image)

        self.idle_timeout = idle_timeout
        di = re.sub(r'\W+','',docker_image)
        self.container_name = f"prover-session-{di}-{uuid.uuid4().hex[:8]}"
        #self.container_name = f"prover-session-{di}-shared"
        self.tmpfs_path_host = os.path.join("/dev/shm", self.container_name)
        self.tmpfs_path_container = "/dev/shm/" + self.container_name
        self.is_setup = False
        self._timer = None
        self._lock = threading.Lock() # For security in critical operations
        self._lock = contextlib.nullcontext()
        self.native_paths = {}
        for prover in provers_to_check:
            path = shutil.which(prover)
            if path:
                self.native_paths[prover] = path

    def setup(self):
        """
        This version uses a shared process.
        """
        with self._lock:
            if self.is_setup:
                return

            os.makedirs(self.tmpfs_path_host, exist_ok=True)
            
            check_result = subprocess.run(['udocker', "--allow-root", 'inspect', self.container_name], capture_output=True)
            if check_result.returncode != 0:
                create_cmd = ['udocker', "--allow-root", 'create', f'--name={self.container_name}', self.docker_image]
                result = subprocess.run(create_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    subprocess.run(['udocker',"--allow-root", 'rm', self.container_name], capture_output=True)
                    retry_result = subprocess.run(create_cmd, capture_output=True, text=True)
                    if retry_result.returncode != 0:
                        raise RuntimeError(f"[{self.container_name}] Failed to create container: {retry_result.stderr}")
            
            self.is_setup = True
        
        self._reset_idle_timer()


    def _reset_idle_timer(self):
        """Resets the timer that will stop the container after inactivity."""
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.idle_timeout, self.kill)
        self._timer.daemon = True 
        self._timer.start()

    def run_prover(self, solver, options, tptp_file, timeout=30):
        """
        Executes a solver on a problem file within the container.
        """
        if solver in self.native_paths:
            executable = self.native_paths[solver]
            command = [executable] + options + [tptp_file]
            return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        
        if not self.is_setup: self.setup()
        self._reset_idle_timer()

        if not os.path.exists(tptp_file):
            raise FileNotFoundError(f"Problem file '{tptp_file}' not found.")

        problem_filename = os.path.basename(tptp_file)
        problem_path_host = os.path.join(self.tmpfs_path_host, problem_filename)
        problem_path_container = os.path.join(self.tmpfs_path_container, problem_filename)

        with open(tptp_file, 'r') as f_in, open(problem_path_host, 'w') as f_out:
            f_out.write(f_in.read())
            
        command = [solver] + options + [problem_path_container]
        
        try:
            result = subprocess.run(
                ["udocker", "--allow-root", "run", f"--volume={self.tmpfs_path_host}:{self.tmpfs_path_container}", self.container_name] + command,
                capture_output=True, text=True, timeout=timeout
            )
        except Exception as e:
            raise TimeoutError

        os.remove(problem_path_host) 
        return result

    def run_agint(self, input_string, timeout=30):
        """
        Executes AGInTRater by passing it a string as standard input.
        """
        if not self.is_setup: self.setup()
        self._reset_idle_timer()
        
        command = ["AGInTRater", "-c"]

        result = subprocess.run(
            ["udocker", "--allow-root", "run", self.container_name] + command,
            input=input_string, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout

    def kill(self):
        """Stops and removes the container and cleans up associated resources."""
        with self._lock:
            if not self.is_setup:
                return

            if self._timer:
                self._timer.cancel()

            subprocess.run(['udocker',"--allow-root", 'rm', '-f', self.container_name], capture_output=True)
            
            # Cleanup of the shared folder
            if os.path.exists(self.tmpfs_path_host):
                for f in os.listdir(self.tmpfs_path_host):
                    os.remove(os.path.join(self.tmpfs_path_host, f))
                os.rmdir(self.tmpfs_path_host)
            
            self.is_setup = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill()
        
    def __del__(self):
        self.kill()


prover_session = Embeded_process()