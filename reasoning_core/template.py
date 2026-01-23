import wrapt
import time
import functools
import pickle, base64
from io import BytesIO
from easydict import EasyDict as edict
from collections import Counter
from collections.abc import Mapping
from reasoning_gym.dataset import ProceduralDataset
from dataclasses import dataclass, fields, field, asdict
from typing import Any
from types import SimpleNamespace
import random
import copy
import signal
from inflection import underscore
import tiktoken

_REGISTRY = dict()



def serialize(data):
    def parquet_friendly(x):
        try:
            pd.DataFrame([x]).to_parquet(BytesIO(), index=False)
            return True
        except:
            return False

    return data if parquet_friendly(data) else base64.b64encode(pickle.dumps(data)).decode()

def deserialize(s):
    def looks_base64(x):
        try:
            return base64.b64encode(base64.b64decode(x)) == x.encode()
        except:
            return False

    return pickle.loads(base64.b64decode(s.encode())) if isinstance(s, str) and looks_base64(s) else s


def seed():
    import random
    random.seed()
    np.random.seed()


def timeout_retry(seconds=10, attempts=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                #raise TimeoutError(f"Timed out after {seconds}s")
                pass
            
            for attempt in range(1, attempts + 1):
                try:
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(seconds)
                    result = func(*args, **kwargs)
                    signal.alarm(0)
                    return result
                except Exception as e:
                    signal.alarm(0)
                    if attempt == attempts:
                        raise
                    time.sleep(0.5)
            
        return wrapper
    return decorator

import psutil  # pip install psutil

class TimeoutException(BaseException): pass

def timeout_retry(seconds=10, attempts=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException()

            for attempt in range(1, attempts + 1):
                old_handler = signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)
                    return result
                except (TimeoutException, Exception) as e:
                    signal.alarm(0) # Ensure alarm is off
                    
                    # --- CRITICAL: Kill external subprocesses (vampire/udocker) ---
                    try:
                        # Find all children spawned by this process and kill them
                        children = psutil.Process().children(recursive=True)
                        for child in children:
                            child.kill()
                        psutil.wait_procs(children, timeout=1)
                    except: pass 
                    # --------------------------------------------------------------

                    if attempt == attempts:
                        raise e  # Re-raise the last exception if out of retries
                    time.sleep(0.5)
                finally:
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
    
    def to_dict(self):
        return {
            'prompt': self.prompt,
            'answer': self.answer,
            'metadata': self.metadata,
            'task': self.task,
        }
        
    @classmethod
    def from_dict(cls, d):
        data = deserialize(d["data"])
        return cls(data=data, answer=d.get("answer"), meta=d.get("meta"), task=d.get("task"))
        
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
    

class Task(ProceduralDataset):
    def __init_subclass__(cls):
        cls.task_name = getattr(cls, 'task_name', prepr_task_name(cls.__name__))
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
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def generate(self):
        raise NotImplementedError 
        """To override, return one problem"""
        return Problem(metadata=edict(), answer="")
        
    def prompt(self,metadata):
        """To override, turns a problem metadata into a prompt"""
        return ""

    def default_score_answer(self, answer, entry):
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
        score = [self.score_answer(y.answer, x) for y in ys]
        assert set(score)!={1}, "The scoring function must return values other than 1 for other answers"
        assert {self.score_answer(y.answer,y)==1 for y in ys}=={True}, "The generated answer must be correct"

        self.score_answer('reajrjrje9595!',x) # should not error out
        self.score_answer('',x) # should not error out
        self.score_answer('import fakemodule',x) # should not eval strings 

        self.generate_example()
        r1=random.random()
        self.generate_example()
        r2=random.random()
        assert r1!=r2
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
        



    def generate_example(self, level=None, max_tokens=8192, **kwargs):
        level = level or getattr(self.config, 'level', 0)
        self.timeout = int(self.base_timeout * (1+level))
        @timeout_retry(self.timeout)
        def inner():
            t0=time.time()
            if level:
                self.config.set_level(level)
            for _ in range(1_000):
                problem = self.generate(**kwargs)
                if problem is None:
                    continue
                problem.prompt = self.prompt(problem.metadata)

                prompt_tokens = len(self.tokenizer.encode(problem.prompt))
                cot_tokens = len(self.tokenizer.encode(problem.metadata.get('cot','') + problem.answer))
                if max_tokens and prompt_tokens > max_tokens:
                    continue
                if max_tokens and cot_tokens > max_tokens:
                    continue
                break  
            
            problem.task = self.task_name

            problem.metadata = edict(problem.metadata)
            problem.metadata['_time']  = time.time() - t0
            problem.metadata['_task']  = problem.task 
            problem.metadata['_level'] = self.config.level
            problem.metadata['_config'] = self.config.to_dict()
            problem.metadata['_prompt_tokens'] = prompt_tokens
            problem.metadata['_cot_tokens'] = cot_tokens

            problem.balancing_key = self.balancing_key(problem)
            problem.deduplication_key = self.deduplication_key(problem)
            return problem
        return inner()

    def generate_balanced_batch(self, batch_size=32, deduplication = False, **kwargs):
        max_per_key = int(batch_size * self.balancing_key_ratio)
        counts = Counter()
        if deduplication:
            deduplication_values = []
        batch = []
        while len(batch) < batch_size:
            ex = self.generate_example(**kwargs)
            b_key = ex.balancing_key
            d_key = ex.deduplication_key
            if d_key is not None and deduplication:
                if d_key in deduplication_values:
                    continue
            if b_key is None or counts[b_key] < max_per_key:
                batch.append(ex)
                if d_key is not None and deduplication:
                    deduplication_values.append(d_key)
                if b_key is not None:
                    counts[b_key] += 1
        return batch


    def __getitem__(self, idx: int) -> dict:
        if self.seed:
            rng = random.Random(self.seed + idx)
        example=self.generate_example()
        example['metadata']['source_dataset'] = example.task

        return {
            "question": example.prompt,
            "answer": example.answer,
            "metadata": example.metadata
            }
        


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

class Reward(wrapt.ObjectProxy):
    def __init__(self, wrapped, tag=None, **kwargs):
        super().__init__(wrapped)
        self._self_annotations = {'tag':tag, **kwargs}

    def __getattr__(self, name):
        if name == "_self_annotations":
            return super().__getattr__(name)
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

