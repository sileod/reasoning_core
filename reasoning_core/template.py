import wrapt
import time
import functools
import pickle, base64
import threading
import subprocess
import warnings
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
from inflection import underscore
import tiktoken
import psutil
from tqdm.auto import tqdm 
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

#template.py

_REGISTRY = dict()


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
                        signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator



class Problem(Mapping):
    def __init__(self, metadata, answer=None, cot=None):
        self.metadata = edict(metadata)
        self.answer = answer
        self.prompt = None
        self.task = self.metadata.get('task', None)
        if cot is not None and self.metadata.cot is None:
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
        
def register_dataset(name, dataset_cls):
    _REGISTRY[name] = dataset_cls


def prepr_task_name(name):
    return underscore(name)


@functools.lru_cache(maxsize=1)
def _load_tokenizer():
    class _WhitespaceTokenizerFallback:
        """Minimal tokenizer fallback when tiktoken assets are unavailable."""
        def encode(self, text):
            return str(text).split()

    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return _WhitespaceTokenizerFallback()
    

class Task(ProceduralDataset):
    def __init_subclass__(cls):
        cls.task_name = getattr(cls, 'task_name', prepr_task_name(cls.__name__))
        cls.category_name = getattr(cls, 'category_name', cls.__module__.split('.')[-1])
        register_dataset(cls.task_name, cls)


    def __init__(self, config=dict(), timeout=10, seed=None, _level=0, *a, **kwa):
        self.seed = seed
        self.config=copy.deepcopy(config)
        self.timeout = timeout
        self.base_timeout = timeout
        self.cls_name = self.__class__.__name__
        self.task_name = prepr_task_name(self.__class__.task_name)
        for k,v in kwa.items():
            setattr(self.config, k, v)
        self.balancing_key_ratio = 0.5
        self.tokenizer = _load_tokenizer()

    def generate(self):
        """To override, return one problem"""
        #return Problem(metadata=edict(), answer="")
        raise NotImplementedError 

        
    def prompt(self,metadata):
        """To override, turns a problem metadata into a prompt"""
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
    
    def validate(self, n_samples=10):
        """Sanity checks to ensure that generation and scoring are working as expected."""
        x=self.generate_example()
        assert isinstance(x, Problem), f"Generated example must be of type Problem, got {type(x)}"
        assert self.score_answer(x.answer, x)==1, "The generated answer must be correct"
        assert x.prompt, "Generated example must have a non-empty prompt"
        ys=[self.generate_example() for _ in range(n_samples)]
        assert len({y.prompt for y in ys})!=1 or n_samples==1, "Examples should not be identical"
        score = [self.score_answer(y.answer, x) for y in ys]
        assert set(score)!={1}, "The scoring function must return values other than 1 for other answers"
        assert {self.score_answer(y.answer,y)==1 for y in ys}=={True}, "The generated answer must be correct"

        # Serialization round-trip smoke test
        rt = copy.copy(x)
        rt.metadata = deserialize(serialize(dict(x.metadata)))
        assert self.score_answer(x.answer, rt) == 1, "score_answer must survive serialize/deserialize round-trip"
        
        self.score_answer('reajrjrje9595!',x) # should not error out
        self.score_answer('',x) # should not error out
        self.score_answer('import fakemodule',x) # should not eval strings 

        c0=copy.deepcopy(self.config)
        self.config.set_level(1)
        assert self.config!=c0
        self.config.set_level(0)
        #assert self.config==c0
        
        self.generate_example()
        r1=random.random()
        self.generate_example()
        r2=random.random()
        assert r1!=r2, "Example generation should not set a seed"


        return ys

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

    def generate_example(self, level=None, max_tokens=8192, **kwargs):
        self.timeout = int(self.base_timeout * (1+level)) if level else int(self.base_timeout)
        @timeout_retry(self.timeout)
        def inner():
            t0=time.time()
            if level:
                self.config.set_level(level)
            with self._override_config(**kwargs) as generate_kwargs:
                for _ in range(1_000):
                    problem = self.generate(**generate_kwargs)
                    if problem is None:
                        continue
                    problem.prompt = self.prompt(problem.metadata)

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

                problem.balancing_key = self.balancing_key(problem)
                problem.deduplication_key = self.deduplication_key(problem)
                return problem
        return inner()

    def generate_balanced_batch(self, batch_size=32, deduplication=False,
                                progress=False, workers=1, **kwargs):
        max_per_key = math.ceil(batch_size * self.balancing_key_ratio)
        counts, seen, batch = Counter(), set(), []

        def try_accept(ex):
            b, d = ex.balancing_key, ex.deduplication_key
            if (deduplication and d in seen) or (b is not None and counts[b] >= max_per_key):
                return False
            batch.append(ex)
            if b is not None: counts[b] += 1
            if deduplication and d is not None: seen.add(d)
            return True

        with tqdm(total=batch_size, disable=not progress) as pbar:
            if workers == 1:
                while len(batch) < batch_size:
                    if try_accept(self.generate_example(**kwargs)): pbar.update(1)
            else:
                submit = lambda pool: pool.submit(self.generate_example, **kwargs)
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    pending = {submit(pool) for _ in range(workers)}
                    while len(batch) < batch_size:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in done:
                            if len(batch) >= batch_size: break
                            if try_accept(f.result()): pbar.update(1)
                        pending |= {submit(pool) for _ in range(workers - len(pending))}
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
    and implement a natural `update()` method (e.g., `self.n_ex += self.c`).
    The base class handles all rounding logic automatically.
    """
    c: float = 1.0
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
                    local_rng = random.Random(object.__getattribute__(self, 'seed'))
                    floor_val = int(float_val)
                    return floor_val + (1 if local_rng.random() < (float_val - floor_val) else 0)
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

    def set_level(self, i: int):
        current_c = self.c
        current_seed = self.seed
        self.__dict__.update(copy.deepcopy(self._base_config_dict))
        self._unrounded = copy.deepcopy(self._base_unrounded)
        self.c = current_c
        self.seed = current_seed
        # Set the flag to enable deterministic updates.
        object.__setattr__(self, '_is_updating', True)
        try:
            object.__setattr__(self, 'level', i)             
            for _ in range(i):
                self.update(self.c)
        finally:
            # Always reset the flag, even if update fails.
            object.__setattr__(self, '_is_updating', False)
        
        object.__setattr__(self, 'level', i) 
        return self

    def update(self, c):
        raise NotImplementedError("Config subclasses must implement 'update'")

    def to_dict(self):
        return asdict(self)

    def __repr__(self) -> str:
        field_strings = []
        for f in fields(self):
            value = getattr(self, f.name)
            field_strings.append(f"{f.name}={value!r}")
        
        return f"{self.__class__.__name__}({', '.join(field_strings)})"

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

