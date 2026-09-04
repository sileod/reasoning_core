import wrapt
import json
import time
from easydict import EasyDict as edict
from collections import Counter
from collections.abc import Mapping
try:
    from reasoning_gym.dataset import ProceduralDataset
except ImportError:
    ProceduralDataset = object
from dataclasses import dataclass, fields, field, asdict
import random
import copy
import math
from difflib import SequenceMatcher
from contextlib import contextmanager
from contextvars import ContextVar
import xxhash
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from .registry import _REGISTRY, prepr_task_name, register_dataset
from .runtime import (
    TimeoutException,
    _stable_dump,
    _strip_docstrings,
    deserialize,
    generator_commit as _generator_commit,
    generator_version as _generator_version,
    load_tokenizer as _load_tokenizer,
    module_behavior_hash as _module_behavior_hash,
    serialize,
    timeout_retry,
    validation_store as _validation_store,
)


def tqdm(iterable=None, *a, **k):
    """tqdm if installed, otherwise a transparent pass-through."""
    try:
        from tqdm.auto import tqdm as _t
    except ImportError:
        return iterable if iterable is not None else _NullBar()
    return _t(iterable, *a, **k) if iterable is not None else _t(*a, **k)


class _NullBar:
    def update(self, *a, **k): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False



_ROUNDING_SEED = ContextVar("reasoning_core_rounding_seed", default=None)
_ROUNDING_SEED_UNSET = object()


def seed():
    import random
    import numpy as np
    random.seed()
    np.random.seed()


def stochastic_rounding(value, seed=_ROUNDING_SEED_UNSET):
    """Explicitly round a float to a nearby int."""
    if seed is _ROUNDING_SEED_UNSET:
        seed = _ROUNDING_SEED.get()
    floor_val = int(value)
    return floor_val + (1 if random.Random(seed).random() < (value - floor_val) else 0)




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
        return iter(self.to_dict())
    def keys(self):
        return self.to_dict().keys()
    def __len__(self):
        return len(self.to_dict())


Problem = Entry


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


class Task(ProceduralDataset):
    config_cls = None
    _distractor_reservoir_size = 64
    _distractor_saturation_patience = 8

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
        self._answer_reservoir = []

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

    def distractor_candidates(self, entry):
        """Yield task-specific plausible wrong answers in canonical answer format."""
        return ()

    def _remember_answer(self, answer):
        if answer is None:
            return
        answer = str(answer)
        if answer in self._answer_reservoir:
            return
        self._answer_reservoir.append(answer)
        del self._answer_reservoir[:-self._distractor_reservoir_size]

    @staticmethod
    def _distractor_similarity(candidate, gold):
        """Cheap surface similarity; correctness belongs exclusively to score_answer."""
        candidate, gold = str(candidate), str(gold)
        char_similarity = SequenceMatcher(None, candidate, gold, autojunk=False).ratio()

        def ngrams(text, n=3):
            if len(text) < n:
                return {text} if text else set()
            return {text[i:i + n] for i in range(len(text) - n + 1)}

        candidate_ngrams, gold_ngrams = ngrams(candidate), ngrams(gold)
        union = candidate_ngrams | gold_ngrams
        ngram_similarity = (
            len(candidate_ngrams & gold_ngrams) / len(union) if union else 1.0
        )
        longest = max(len(candidate), len(gold))
        length_similarity = min(len(candidate), len(gold)) / longest if longest else 1.0
        return 0.55 * char_similarity + 0.35 * ngram_similarity + 0.10 * length_similarity

    def generate_distractors(self, entry, n=16, max_candidates=64):
        """Return up to ``n`` distinct, scorer-validated wrong answers.

        Task-provided candidates are preferred. Observed answers from this task and,
        when needed, newly generated same-task answers provide a bounded fallback.
        """
        n, max_candidates = max(0, int(n)), max(0, int(max_candidates))
        if not n or not max_candidates:
            return []

        inspected = 0
        valid_candidates = {}
        seen_candidates = set()

        def inspect(values):
            nonlocal inspected
            values = iter(values)
            while inspected < max_candidates:
                try:
                    value = next(values)
                except StopIteration:
                    break
                inspected += 1
                if value is None:
                    continue
                candidate = str(value)
                if candidate in seen_candidates:
                    continue
                seen_candidates.add(candidate)
                try:
                    valid = self.score_answer(candidate, entry) < 1
                except Exception:
                    valid = False
                if valid:
                    valid_candidates[candidate] = len(valid_candidates)

        inspect(self.distractor_candidates(entry))
        if inspected < max_candidates and len(valid_candidates) < n:
            inspect(tuple(self._answer_reservoir))

        # Repeated answers indicate a small observed vocabulary. Stop once that
        # vocabulary saturates instead of spending the whole budget on duplicates.
        stale = 0
        while inspected < max_candidates and len(valid_candidates) < n:
            before = tuple(self._answer_reservoir)
            try:
                generated = self.generate_example()
            except Exception:
                break
            inspect((generated.answer,))
            stale = stale + 1 if tuple(self._answer_reservoir) == before else 0
            if stale >= self._distractor_saturation_patience:
                break

        gold = str(entry.answer)
        ranked = sorted(
            valid_candidates,
            key=lambda candidate: (
                -self._distractor_similarity(candidate, gold),
                valid_candidates[candidate],
            ),
        )
        return ranked[:n]
        
    def __call__(self, *args, **kwargs):
        return self.generate_example(*args, **kwargs)
    
    def validate(self, n_samples=10, cache=False, refresh=False):
        """Smoke tests to ensure that generation and scoring are working as expected."""
        summary = getattr(type(self), "summary", None)
        assert isinstance(summary, str) and summary.strip(), (
            f"{type(self).__name__}.summary must be a class-level one-line coverage spec")
        assert summary == summary.strip() and "\n" not in summary and "\r" not in summary, (
            f"{type(self).__name__}.summary must be one trimmed line")
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

        # Strict JSON round-trip: production writers do not provide a fallback encoder.
        rt = copy.copy(x)
        rt.metadata = edict(json.loads(json.dumps(dict(x.metadata))))
        assert self.score_answer(x.answer, rt) == 1, "score_answer must survive serialize/deserialize round-trip"
        from reasoning_core import score_answer as dispatch_score
        wire = edict({**x.to_dict(), "metadata": json.dumps(dict(x.metadata))})
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
        try:
            json.dumps(dict(x.metadata))
        except TypeError as bad:
            # Still a TypeError, which is what the writer raises and what callers catch.
            # Only the message changes: raised bare this said "Object of type Entity is
            # not JSON serializable" and named neither metadata nor the task, and the
            # writer has no fallback encoder, so such a task fails mid-collection.
            raise TypeError(
                f"{type(self).__name__}.metadata must be JSON-serializable: {bad}. "
                "Store plain values and keep helper objects out of metadata.") from bad
        assert self.score_answer(x.answer, x)==1, "The generated answer must be correct"
        assert x.prompt, "Generated example must have a non-empty prompt"
        assert len({y.prompt for y in ys})!=1 or n_samples==1, "Examples should not be identical"
        other_answers = [y.answer for y in ys if y.answer != x.answer]
        score = [self.score_answer(answer, x) for answer in other_answers]
        assert not score or set(score) != {1}, (
            "score_answer must return values other than 1 for other answers"
        )
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
        Return the canonical content used to deduplicate examples.

        By default this is the full prompt/answer pair.  For payload-rendered prompts,
        only the shallow payload order is normalized; the surrounding instructions
        remain part of the key.  Tasks may override this with stronger, domain-safe
        invariances.
        """
        prompt = str(problem.prompt)
        payload = problem.metadata.get("payload")
        if isinstance(payload, Mapping):
            rendered = render_payload(payload)
            if rendered and prompt.count(rendered) == 1:
                normalized = render_payload({key: payload[key] for key in sorted(payload)})
                prompt = prompt.replace(rendered, normalized, 1)
        return json.dumps(
            {"prompt": prompt, "answer": problem.answer},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        )

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
                    # `_answer_tokens` must be the ANSWER only. It used to be cot+answer, which made
                    # 9 CoT-bearing tasks look like long-answer tasks (logic_nli read 122 tokens for
                    # the answer "Yes") and silently corrupted any length analysis built on it. The
                    # generation FILTER below still uses cot+answer, so which examples are kept is
                    # unchanged -- only the reported metric is fixed.
                    _cot = problem.metadata.get('cot','') or ''
                    cot_tokens = len(self.tokenizer.encode(_cot))
                    answer_tokens = len(self.tokenizer.encode(problem.answer))
                    # the FILTER keeps the original expression verbatim -- tok(cot+answer) is not
                    # tok(cot)+tok(answer) (tokenizers merge across the join), so recomputing it from
                    # the parts would shift the threshold and change which examples are kept.
                    if max_tokens and prompt_tokens > max_tokens:
                        continue
                    if max_tokens and len(self.tokenizer.encode(_cot + problem.answer)) > max_tokens:
                        continue
                    break
                else:
                    raise RuntimeError(
                        f"{self.task_name}: failed to generate an admissible example "
                        "after 1000 attempts"
                    )
                
                problem.task = self.task_name

                problem.metadata = edict(problem.metadata)
                problem.metadata['_time']  = time.time() - t0
                problem.metadata['_task']  = problem.task 
                problem.metadata['_level'] = self.config.level
                problem.metadata['_config'] = self.config.to_dict()
                problem.metadata['_prompt_tokens'] = prompt_tokens
                problem.metadata['_answer_tokens'] = answer_tokens
                problem.metadata['_cot_tokens'] = cot_tokens
                generator_name = self.__class__.__module__.split(".", 1)[0]
                problem.metadata['_generator_name'] = generator_name
                problem.metadata['_generator_version'] = _generator_version(generator_name)
                problem.metadata['_generator_commit'] = _generator_commit()
                problem.metadata['_task_version'] = getattr(self, "task_version", "0")
                problem.metadata['_task_behavior_hash'] = self.behavior_hash()

                problem.balancing_key = self.balancing_key(problem)
                canonical = self.deduplication_key(problem)
                problem.deduplication_key = canonical
                problem.metadata["_deduplication_key"] = (
                    None if canonical is None
                    else xxhash.xxh3_128_hexdigest(
                        canonical if isinstance(canonical, (str, bytes))
                        else json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), default=str)
                    )
                )
                self._remember_answer(problem.answer)
                return problem
        return inner()

    def generate_examples(self, **kwargs):
        """Generate one atomic group for balanced batching."""
        return [self.generate_example(**kwargs)]

    def generate_balanced_batch(self, batch_size=32, deduplication=False,
                                progress=False, workers=1, **kwargs):
        max_per_key = math.ceil(batch_size * self.balancing_key_ratio)
        counts, seen, batch = Counter(), set(), []
        # A batch is fillable only if (distinct keys) * max_per_key >= batch_size, i.e. roughly
        # balancing_key_ratio >= 1/(keys the task can actually realise). When it is not, the fill
        # loop below never terminates -- table_qa burned ~3 CPU-hours that way at level 0, where the
        # generator can only produce 2 keys against a 0.25 cap. Infeasibility cannot be detected
        # before sampling (the realised key set is unknown a priori), so count attempts and raise
        # with the observed distribution instead of spinning.
        attempts = [0]
        budget = max(200, 40 * batch_size)

        def try_accept(group):
            attempts[0] += 1
            if attempts[0] > budget and len(batch) < batch_size:
                need = math.ceil(1 / max(len(counts), 1) * 100) / 100
                raise RuntimeError(
                    f"{type(self).__name__}.generate_balanced_batch could not fill {batch_size} rows "
                    f"in {attempts[0]} attempts (got {len(batch)}). Only {len(counts)} distinct "
                    f"balancing keys were realised, so at most {len(counts)}*{max_per_key}="
                    f"{len(counts) * max_per_key} rows are reachable. Raise balancing_key_ratio to "
                    f">= {need} (it is {self.balancing_key_ratio}), widen balancing_key, or make the "
                    f"generator reach more keys. Observed: {dict(counts)}")
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
    """Base task config with resettable difficulty levels."""
    level: int = 0
    seed: int = None
    size: int = None

    def __post_init__(self):
        self._base_config_dict = copy.deepcopy(self.__dict__)
        if self.level > 0:
            initial_level = self.level
            self.set_level(initial_level)

    def _apply_difficulty_level(self, i: int, apply):
        current_seed = self.seed
        self.__dict__.update(copy.deepcopy(self._base_config_dict))
        self.seed = current_seed
        rounding_seed_token = _ROUNDING_SEED.set(current_seed)
        try:
            object.__setattr__(self, 'level', i)             
            apply(i)
        finally:
            _ROUNDING_SEED.reset(rounding_seed_token)
        
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
