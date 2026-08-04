import wrapt
import json
import time
import functools
import ast
import hashlib
import pickle, base64
import threading
import subprocess
import warnings
import sys
from easydict import EasyDict as edict
from collections import Counter
from collections.abc import Mapping
try:
    from reasoning_gym.dataset import ProceduralDataset
except ImportError:
    ProceduralDataset = object
from dataclasses import dataclass, fields, field, asdict
from typing import Any
from types import SimpleNamespace
import random
import copy
import math
import signal
from contextlib import contextmanager
from contextvars import ContextVar
from inflection import underscore
import tiktoken
from appdirs import user_cache_dir
import os
import psutil
from tqdm.auto import tqdm 
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

#template.py

_REGISTRY = dict()
_ROUNDING_SEED = ContextVar("reasoning_core_rounding_seed", default=None)
_ROUNDING_SEED_UNSET = object()


@functools.lru_cache(maxsize=None)
def _generator_version(package_name):
    module = sys.modules.get(package_name)
    version = getattr(module, "__version__", None)
    if version:
        return version

    try:
        from importlib.metadata import version as package_version

        return package_version(package_name)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _generator_commit():
    for name in ("SOURCE_COMMIT", "GIT_COMMIT", "COMMIT_SHA", "GITHUB_SHA"):
        value = os.environ.get(name)
        if value:
            return value

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        ).strip()
    except Exception:
        return None


def _strip_docstrings(node):
    node = copy.deepcopy(node)
    for n in ast.walk(node):
        body = getattr(n, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr) and
                isinstance(body[0].value, ast.Constant) and
                isinstance(body[0].value.value, str)):
            del body[0]
    return node


@functools.lru_cache(maxsize=None)
def _module_behavior_hash(module_name):
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if not path or not path.endswith(".py"):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        canonical = ast.dump(_strip_docstrings(tree), include_attributes=False)
        return hashlib.sha1(canonical.encode()).hexdigest()[:16]
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _validation_store():
    from nfsdict import NfsDict
    base_dir = os.environ.get("RC_VALIDATE_CACHE_DIR",
                              os.path.join(user_cache_dir("reasoning_core"), "validation"))
    return NfsDict(name="examples", base_dir=base_dir, serializer="json")


def _parquet_safe(x):
    import pandas as pd
    from io import BytesIO
    try:
        pd.DataFrame([x]).to_parquet(BytesIO(), index=False)
        return True
    except Exception:
        return False

def serialize(data):
    if _parquet_safe(data):
        return data
    return "b64:" + base64.b64encode(pickle.dumps(data)).decode()

def deserialize(s):
    if isinstance(s, str) and s.startswith("b64:"):
        return pickle.loads(base64.b64decode(s[4:].encode()))
    return s


def seed():
    import random
    import numpy as np
    random.seed()
    np.random.seed()


def stochastic_rounding(value, seed=_ROUNDING_SEED_UNSET):
    """Round a float to a nearby int using the shared Config rounding rule."""
    if seed is _ROUNDING_SEED_UNSET:
        seed = _ROUNDING_SEED.get()
    floor_val = int(value)
    return floor_val + (1 if random.Random(seed).random() < (value - floor_val) else 0)




class TimeoutException(BaseException): pass

_RETRYABLE = (TimeoutException, subprocess.SubprocessError)

def timeout_retry(seconds=15, attempts=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            on_main = threading.current_thread() is threading.main_thread()
            if not on_main:
                warnings.warn(
                    "timeout_retry: signal-based timeout unavailable off the main thread; "
                    "call will run without a timeout guard.",
                    stacklevel=3,
                )

            def handler(signum, frame):
                raise TimeoutException()

            for attempt in range(1, attempts + 1):
                if on_main:
                    old_handler = signal.signal(signal.SIGALRM, handler)
                    signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                    if on_main:
                        signal.alarm(0)
                    return result
                except _RETRYABLE as e:
                    if on_main:
                        signal.alarm(0)
                    
                    # --- CRITICAL: Kill external subprocesses (vampire/udocker) ---
                    try:
                        children = psutil.Process().children(recursive=True)
                        for child in children:
                            child.kill()
                        psutil.wait_procs(children, timeout=1)
                    except: pass 
                    # --------------------------------------------------------------

                    if attempt == attempts:
                        raise e
                    time.sleep(0.5)
                finally:
                    if on_main:
                        signal.alarm(0)   # ALWAYS cancel — else a leaked alarm from a non-retryable
                                          # exception path fires later at an arbitrary point (e.g. inside
                                          # logging), raising TimeoutException(BaseException) past callers'
                                          # `except Exception` and crashing the whole generation loop.
                        signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator



class Entry(Mapping):
    def __init__(self, metadata, answer=None, cot=None):
        self.metadata = edict(metadata)
        self.answer = answer
        self.prompt = None
        self.task = self.metadata.get('task', None)
        if cot is not None and self.metadata.get('cot') is None:
            self.metadata.cot = cot
        self.cot= self.metadata.get('cot','')
        
    def to_dict(self):
        return {
            'prompt': self.prompt,
            'answer': self.answer,
            'metadata': self.metadata,
            'task': self.task,
            'cot': self.metadata.get('cot','')
        }
        
    @classmethod
    def from_dict(cls, d):
        metadata = deserialize(d.get("metadata", d.get("data", {})))
        return cls(metadata=metadata, answer=d.get("answer"), cot=d.get("cot"))
        
    def __repr__(self):
        s=""
        for k,v in self.to_dict().items():
            s+=f"---{k.title()}:{v}\n"
        return s
        
    __str__=__repr__

    def __getitem__(self,k):
        return getattr(self,k)
    def __iter__(self):
        yield from self.to_dict().items()
    def keys(self):
        return self.to_dict().keys()
    def __len__(self):
        return len(self.to_dict())


Problem = Entry
        
def register_dataset(name, dataset_cls):
    _REGISTRY[name] = dataset_cls


def prepr_task_name(name):
    return underscore(name)


def render_payload(payload):
    """Render a JSON-friendly prompt payload mapping as labeled blocks."""
    return "\n\n".join(
        f"{key.replace('_', ' ').title()}:\n{value}"
        for key, value in payload.items()
    )


def _shuffle_payload(payload, p=0.0, seed=None):
    if not p or not isinstance(payload, Mapping) or len(payload) < 2:
        return payload
    rng = random.Random(seed)
    if rng.random() >= p:
        return payload
    keys = list(payload.keys())
    rng.shuffle(keys)
    return {key: payload[key] for key in keys}


class Payload(dict):
    """Backward-compatible wrapper for rendering prompt payload mappings."""

    def __init__(self, *args, randomizable=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomizable = randomizable
        self.original_order = list(self.keys())
        self.order = list(self.original_order)

    def __str__(self):
        return render_payload({key: self[key] for key in self.order})

    def maybe_shuffle(self, p=0.2, seed=None):
        rng = random.Random(seed)
        if self.randomizable and rng.random() < p:
            rng.shuffle(self.order)
            items = [(key, self[key]) for key in self.order]
            self.clear()
            self.update(items)
        return self

    @classmethod
    def maybe_shuffle_mapping(cls, payload, p=0.0):
        return _shuffle_payload(payload, p=p, seed=random.randrange(1 << 32))

    @classmethod
    def maybe_shuffle_metadata(cls, metadata, p=0.0):
        if p and "payload" in metadata:
            metadata.payload = cls.maybe_shuffle_mapping(metadata.payload, p)


@functools.lru_cache(maxsize=1)
def _load_tokenizer():
    cache_dir = user_cache_dir("reasoning_core")  # ~/.cache/reasoning_core on Linux
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", cache_dir)

    class _WhitespaceTokenizerFallback:
        """Minimal tokenizer fallback when tiktoken assets are unavailable."""
        def encode(self, text):
            return str(text).split()

    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return _WhitespaceTokenizerFallback()
    

class Task(ProceduralDataset):
    config_cls = None

    def __init_subclass__(cls):
        cls.task_name = getattr(cls, 'task_name', prepr_task_name(cls.__name__))
        cls.category_name = getattr(cls, 'category_name', cls.__module__.split('.')[-1])
        register_dataset(cls.task_name, cls)


    def __init__(self, config=None, timeout=10, seed=None, _level=0, *a, **kwa):
        self.seed = seed
        if config is None:
            config_cls = self.config_cls or Config
            config = config_cls()
        self.config=copy.deepcopy(config)
        self.timeout = timeout
        self.base_timeout = timeout
        self.cls_name = self.__class__.__name__
        self.task_name = prepr_task_name(self.__class__.task_name)
        for k,v in kwa.items():
            setattr(self.config, k, v)
        self.balancing_key_ratio = 0.5
        self.tokenizer = _load_tokenizer()
        self._config_level_seen = getattr(self.config, "level", None)

    def generate_entry(self):
        """To override in new tasks, return one Entry."""
        if type(self).generate is not Task.generate:
            return self.generate()
        raise NotImplementedError

    def generate(self):
        """Legacy alias for generate_entry()."""
        if type(self).generate_entry is not Task.generate_entry:
            return self.generate_entry()
        raise NotImplementedError

    def render_prompt(self, metadata):
        """To override in new tasks, render entry metadata as a prompt."""
        if type(self).prompt is not Task.prompt:
            return self.prompt(metadata)
        return ""

    def prompt(self, metadata):
        """Legacy alias for render_prompt(metadata)."""
        if type(self).render_prompt is not Task.render_prompt:
            return self.render_prompt(metadata)
        return ""

    def score_answer(self, answer, entry):
        """To override in most cases; entry has entry.metadata and entry.answer fields"""
        reference = entry['answer']
        prepr = lambda x: str(x).strip()
        answer, reference = prepr(answer), prepr(reference)
        if answer==reference:
            return 1
        return 0
        
    def __call__(self, *args, **kwargs):
        return self.generate_example(*args, **kwargs)
    
    def validate(self, n_samples=10, cache=False, refresh=False):
        """Smoke tests to ensure that generation and scoring are working as expected."""
        if cache:
            key = f"{self.task_name}:{self.config.level}"
            signature = {
                "hash": self.behavior_hash(),
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
            }
            record = None if refresh else _validation_store().get(key)
            examples = record.get("examples", []) if record and record.get("signature") == signature else []
            if len(examples) < n_samples + 1:
                examples = [self.generate_example() for _ in range(n_samples + 1)]
                _validation_store()[key] = {
                    "signature": signature,
                    "examples": [ex.to_dict() for ex in examples],
                }
            else:
                restored = []
                for ex in examples[:n_samples + 1]:
                    problem = Entry.from_dict(ex)
                    problem.prompt = ex.get("prompt")
                    restored.append(problem)
                examples = restored
            x, ys = examples[0], examples[1:]
            self._check_validation_examples(x, ys, n_samples)
            return ys

        x = self.generate_example()
        ys = [self.generate_example() for _ in range(n_samples)]
        self._check_validation_examples(x, ys, n_samples)

        # Serialization round-trip smoke test
        rt = copy.copy(x)
        rt.metadata = deserialize(serialize(dict(x.metadata)))
        assert self.score_answer(x.answer, rt) == 1, "score_answer must survive serialize/deserialize round-trip"
        from reasoning_core import score_answer as dispatch_score
        wire = edict({**x.to_dict(), "metadata": json.dumps(dict(x.metadata), default=str)})
        assert dispatch_score(x.answer, wire) == 1, "score_answer must survive JSON metadata dispatch"
        
        self.score_answer('reajrjrje9595!',x) # should not error out
        self.score_answer('',x) # should not error out
        self.score_answer('import fakemodule',x) # should not eval strings 

        c0=copy.deepcopy(self.config)
        self.config.set_level(self.config.level+1)
        assert self.config!=c0
        self.config.set_level(0)
        #assert self.config==c0
        
        self.generate_example()
        r1=random.random()
        self.generate_example()
        r2=random.random()
        assert r1!=r2, "Example generation should not set a seed"

        return ys

    def _check_validation_examples(self, x, ys, n_samples):
        assert isinstance(x, Entry), f"Generated example must be of type Entry, got {type(x)}"
        assert self.score_answer(x.answer, x)==1, "The generated answer must be correct"
        assert x.prompt, "Generated example must have a non-empty prompt"
        assert len({y.prompt for y in ys})!=1 or n_samples==1, "Examples should not be identical"
        score = [self.score_answer(y.answer, x) for y in ys]
        assert set(score)!={1}, "score_answer must return values other than 1 for other answers"
        assert {self.score_answer(y.answer,y)==1 for y in ys}=={True}, "The generated answer must be correct"
        self.score_answer('reajrjrje9595!',x)
        self.score_answer('',x)
        self.score_answer('import fakemodule',x)

    def postprocess_dataset(self, df):
        """to override, apply deduplication and filtering"""
        return df
        
    def balancing_key(self, problem):
        """
        To override, an optional feature that must be limited in fequency.
        This can prevent label inbalance or frequency of easy problems.
        """
        return str(problem.answer)

    def deduplication_key(self, problem):
        """
        To override, an optional feature that must be the key to deduplicate examples.
        This can prevent the generation of the same problem.
        """
        return None

    def on_config_level_change(self):
        pass

    def behavior_hash(self):
        return _module_behavior_hash(self.__class__.__module__)
        



    @contextmanager
    def _override_config(self, **overrides):
        config_dict = self.config.to_dict()
        applicable = {k: v for k, v in overrides.items() if k in config_dict}
        saved = {k: config_dict[k] for k in applicable}
        for k, v in applicable.items():
            setattr(self.config, k, v)
        try:
            yield {k: v for k, v in overrides.items() if k not in config_dict}
        finally:
            for k, v in saved.items():
                setattr(self.config, k, v)

    def generate_example(self, level=None, max_tokens=8192, payload_shuffle_prob=0.0, **kwargs):
        self.timeout = int(self.base_timeout * (1+level)) if level else int(self.base_timeout)
        @timeout_retry(self.timeout)
        def inner():
            t0=time.time()
            if level is not None:
                self.config.set_level(level)
                if self.config.level != self._config_level_seen:
                    self.on_config_level_change()
                    self._config_level_seen = self.config.level
            with self._override_config(**kwargs) as generate_kwargs:
                for _ in range(1_000):
                    problem = self.generate_entry(**generate_kwargs)
                    if problem is None:
                        continue
                    if payload_shuffle_prob and "payload" in problem.metadata:
                        problem.metadata.payload = _shuffle_payload(
                            problem.metadata.payload,
                            p=payload_shuffle_prob,
                            seed=random.randrange(1 << 32),
                        )
                    problem.prompt = self.render_prompt(problem.metadata)

                    prompt_tokens = len(self.tokenizer.encode(problem.prompt))
                    answer_tokens = len(self.tokenizer.encode(problem.metadata.get('cot','') + problem.answer))
                    if max_tokens and prompt_tokens > max_tokens:
                        continue
                    if max_tokens and answer_tokens > max_tokens:
                        continue
                    break  
                
                problem.task = self.task_name

                problem.metadata = edict(problem.metadata)
                problem.metadata['_time']  = time.time() - t0
                problem.metadata['_task']  = problem.task 
                problem.metadata['_level'] = self.config.level
                problem.metadata['_config'] = self.config.to_dict()
                problem.metadata['_prompt_tokens'] = prompt_tokens
                problem.metadata['_answer_tokens'] = answer_tokens
                generator_name = self.__class__.__module__.split(".", 1)[0]
                problem.metadata['_generator_name'] = generator_name
                problem.metadata['_generator_version'] = _generator_version(generator_name)
                problem.metadata['_generator_commit'] = _generator_commit()
                problem.metadata['_task_version'] = getattr(self, "task_version", "0")
                problem.metadata['_task_behavior_hash'] = self.behavior_hash()

                problem.balancing_key = self.balancing_key(problem)
                problem.deduplication_key = self.deduplication_key(problem)
                return problem
        return inner()

    def generate_examples(self, **kwargs):
        """Generate one atomic group for balanced batching."""
        return [self.generate_example(**kwargs)]

    def generate_balanced_batch(self, batch_size=32, deduplication=False,
                                progress=False, workers=1, **kwargs):
        max_per_key = math.ceil(batch_size * self.balancing_key_ratio)
        counts, seen, batch = Counter(), set(), []

        def try_accept(group):
            if len(batch) + len(group) > batch_size:
                return 0
            keys = Counter(ex.balancing_key for ex in group if ex.balancing_key is not None)
            dedup_keys = [ex.deduplication_key for ex in group if ex.deduplication_key is not None]
            if any(counts[key] + n > max_per_key for key, n in keys.items()):
                return 0
            if deduplication and (seen.intersection(dedup_keys) or len(dedup_keys) != len(set(dedup_keys))):
                return 0
            batch.extend(group)
            counts.update(keys)
            if deduplication: seen.update(dedup_keys)
            return len(group)

        with tqdm(total=batch_size, disable=not progress) as pbar:
            if workers == 1:
                while len(batch) < batch_size:
                    pbar.update(try_accept(self.generate_examples(**kwargs)))
            else:
                submit = lambda pool: pool.submit(self.generate_examples, **kwargs)
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    pending = {submit(pool) for _ in range(min(workers, batch_size))}
                    while len(batch) < batch_size:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in done:
                            if len(batch) >= batch_size: break
                            pbar.update(try_accept(f.result()))
                        target = min(workers, batch_size - len(batch))
                        pending |= {submit(pool) for _ in range(target - len(pending))}
        return batch


    def __getitem__(self, idx: int) -> dict:
        example=self.generate_example()
        example['metadata']['source_dataset'] = example.task

        return {
            "question": example.prompt,
            "answer": example.answer,
            "metadata": example.metadata
            }
        

class DevTask(Task):
    """Task subclass for development/experimental tasks that won't be auto-registered."""
    def __init_subclass__(cls):
        cls.task_name = getattr(cls, 'task_name', prepr_task_name(cls.__name__))
        # Don't call register_dataset - skip auto-registration


@dataclass
class Config:
    """
    Base config providing transparent stochastic rounding.

    A subclass only needs to define its attributes with `int` type hints
    and implement `apply_difficulty(level)`.
    The base class handles all rounding logic automatically.
    """
    level: int = 0
    seed: int = None
    size: int = None

    def __post_init__(self):
        # This flag is the key to differentiating behavior during updates.
        object.__setattr__(self, '_is_updating', False)
        
        self._unrounded = SimpleNamespace()

        self._stochastic_fields = {
            f.name for f in fields(self) 
            if f.type is int and not f.name.startswith('_') and f.name not in ['level', 'size', 'seed']
        }
        for name in self._stochastic_fields:
            if name in self.__dict__:
                setattr(self._unrounded, name, float(self.__dict__.pop(name)))
        
        # Save the base state before any level-based updates are applied.
        self._base_unrounded = copy.deepcopy(self._unrounded)
        self._base_config_dict = copy.deepcopy(self.__dict__)

        # Apply updates if initialized with level > 0.
        if self.level > 0:
            # We need to capture the level passed to __init__ before calling set_level,
            # as set_level will reset it.
            initial_level = self.level
            # Use the existing set_level logic to apply the updates.
            # This is clean and avoids duplicating code.
            self.set_level(initial_level)

    def __getattribute__(self, name: str) -> Any:
        try:
            stochastic_fields = object.__getattribute__(self, '_stochastic_fields')
            if name in stochastic_fields:
                is_updating = object.__getattribute__(self, '_is_updating')
                float_val = getattr(object.__getattribute__(self, '_unrounded'), name)
                
                # If updating, return the raw float for deterministic calculations.
                # Otherwise, return the stochastically rounded value.
                if is_updating:
                    return float_val
                else:
                    return stochastic_rounding(float_val, object.__getattribute__(self, 'seed'))
        except AttributeError:
            pass # Object is still initializing.
            
        return object.__getattribute__(self, name)

    def get_true_value(self, name: str) -> float:
        """Returns the unrounded float value of a stochastic field."""
        if name in self._stochastic_fields:
            return getattr(self._unrounded, name)
        return getattr(self, name)

    def __setattr__(self, name: str, value: Any):
        try:
            if name in object.__getattribute__(self, '_stochastic_fields'):
                setattr(object.__getattribute__(self, '_unrounded'), name, float(value))
                return
        except AttributeError:
            pass # Object is still initializing.
            
        object.__setattr__(self, name, value)

    def _apply_difficulty_level(self, i: int, apply):
        current_seed = self.seed
        self.__dict__.update(copy.deepcopy(self._base_config_dict))
        self._unrounded = copy.deepcopy(self._base_unrounded)
        self.seed = current_seed
        # Set the flag to enable deterministic updates.
        object.__setattr__(self, '_is_updating', True)
        rounding_seed_token = _ROUNDING_SEED.set(current_seed)
        try:
            object.__setattr__(self, 'level', i)             
            apply(i)
        finally:
            _ROUNDING_SEED.reset(rounding_seed_token)
            # Always reset the flag, even if update fails.
            object.__setattr__(self, '_is_updating', False)
        
        object.__setattr__(self, 'level', i) 
        return self

    def set_level(self, i: int):
        return self._apply_difficulty_level(i, self.apply_difficulty)

    def apply_difficulty(self, level: int):
        """Apply the target difficulty level from the base config state.

        Subclasses should override this with an explicit non-recursive formula.
        The default preserves legacy behavior by replaying `update(1)`.
        """
        for _ in range(level):
            self.update(1)

    def update(self, c):
        raise NotImplementedError("Config subclasses must implement 'update'")

    def to_unrounded_dict(self):
        result = {}
        for f in fields(self):
            if f.name in self._stochastic_fields:
                result[f.name] = self.get_true_value(f.name)
            else:
                result[f.name] = getattr(self, f.name)
        return result

    def to_dict(self):
        return asdict(self)

    def __repr__(self) -> str:
        field_strings = []
        for f in fields(self):
            value = getattr(self, f.name)
            field_strings.append(f"{f.name}={value!r}")
        
        return f"{self.__class__.__name__}({', '.join(field_strings)})"


def assert_difficulty_update_equivalence(config, levels=range(6), seed=0):
    """Assert `apply_difficulty(level)` matches repeated legacy `update(1)`.

    This is intended as a cheap migration test for task configs that add an
    explicit `apply_difficulty` implementation while keeping `update`.
    """
    def equal(a, b):
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        return a == b

    for level in levels:
        via_apply = copy.deepcopy(config)
        via_update = copy.deepcopy(config)
        if getattr(via_apply, "seed", None) is None:
            via_apply.seed = seed
        if getattr(via_update, "seed", None) is None:
            via_update.seed = seed
        via_apply.set_level(level)
        via_update._apply_difficulty_level(
            level,
            lambda target_level, cfg=via_update: Config.apply_difficulty(cfg, target_level),
        )
        apply_state = via_apply.to_dict()
        update_state = via_update.to_dict()
        if apply_state.keys() != update_state.keys() or any(
            not equal(apply_state[k], update_state[k]) for k in apply_state
        ):
            raise AssertionError(
                f"{config.__class__.__name__} difficulty migration differs at level {level}: "
                f"apply_difficulty={apply_state}, repeated_update={update_state}"
            )
    return True

class Reward(wrapt.ObjectProxy):
    def __init__(self, wrapped, tag=None, **kwargs):
        super().__init__(wrapped)
        self._self_annotations = {'tag':tag, **kwargs}

    def __getattr__(self, name):
        if name == "_self_annotations":
            raise AttributeError(name)
        if name in self._self_annotations:
            return self._self_annotations[name]
        return getattr(self.__wrapped__, name)

    def __setattr__(self, name, value):
        if name in ("_self_annotations", "__wrapped__"):
            super().__setattr__(name, value)
        elif name in self._self_annotations:
            self._self_annotations[name] = value
        else:
            setattr(self.__wrapped__, name, value)
