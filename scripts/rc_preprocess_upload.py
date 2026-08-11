#!/usr/bin/env python3
"""NFS-friendly streaming rebuild for reasoning-core/procedural-pile.

This avoids materializing the full Hugging Face Dataset locally. Deduplication is
done with sequential bucket files and a compact keep bitmap:

  pass 1A: stream source rows -> write 128-bit semantic/exact-pair hash + row index buckets
  pass 1B: sort one bucket at a time -> mark first occurrence in bitmap
  pass 2: stream source again -> transform kept rows -> parquet shard -> upload/delete

Smoke test, no upload:
  python scripts/rc_preprocess_upload.py --smoke_test 10000 --dry_run

Full upload:
  python scripts/rc_preprocess_upload.py
"""

import argparse
import contextlib
import gc
import hashlib
import html
import json
import multiprocessing as mp
import os
import random
import shutil
import signal
import struct
import sys
import tempfile
import time
import traceback
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from datasets import Features, Value, disable_caching, load_dataset, load_dataset_builder
from easydict import EasyDict as edict
from huggingface_hub import CommitOperationDelete, HfApi
from tqdm import tqdm

from reasoning_core import get_task, list_tasks, score_answer

disable_caching()


TARGET_FEATURES = Features({
    "task": Value("large_string"),
    "prompt": Value("large_string"),
    "answer": Value("large_string"),
    "metadata": Value("large_string"),
    "cot": Value("large_string"),
    "level": Value("int64"),
    "mode": Value("large_string"),
})

TARGET_SCHEMA = pa.schema([
    ("task", pa.large_string()),
    ("prompt", pa.large_string()),
    ("answer", pa.large_string()),
    ("metadata", pa.large_string()),
    ("cot", pa.large_string()),
    ("level", pa.int64()),
    ("mode", pa.large_string()),
])


def output_schema(dataset_name):
    """Return the published schema, excluding preprocessing-only columns."""
    if dataset_name == "procedural-pile":
        return pa.schema([
            field for field in TARGET_SCHEMA
            if field.name != "cot"
        ])
    return TARGET_SCHEMA

RECORD = struct.Struct("<QQQ")  # hash_lo, hash_hi, row_index
DEDUPLICATION_VERSION = "metadata-key-or-prompt-answer-v1"
_SCORE_DEVNULL = None
_VERIFY_POOLS = None
_VERIFY_ARGS = None


class Timeout(Exception):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def stable_u01(*parts, seed=42):
    h = xxhash.xxh64(seed=seed)
    for part in parts:
        h.update(str(part).encode())
        h.update(b"\0")
    return h.intdigest() / 2**64


def deduplication_hash(row):
    """Prefer generator-provided semantic keys; safely fall back to prompt+answer."""
    key = metadata_obj(row.get("metadata")).get("_deduplication_key")
    if isinstance(key, str) and len(key) == 32:
        try:
            return struct.unpack("<QQ", bytes.fromhex(key))
        except ValueError:
            pass
    canonical = json.dumps(
        {"prompt": str(row.get("prompt", "")), "answer": str(row.get("answer", ""))},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return struct.unpack("<QQ", xxhash.xxh3_128_digest(canonical))


def metadata_obj(metadata):
    if isinstance(metadata, dict):
        return metadata
    try:
        return json.loads(metadata or "{}")
    except Exception:
        return {}


def metadata_level(metadata):
    obj = metadata_obj(metadata)
    try:
        return int(obj.get("_level", obj.get("level", 0)) or 0)
    except Exception:
        return 0


def metadata_task(row, meta):
    task = str(row.get("task") or meta.get("_task") or meta.get("task") or "")
    if task == "reasoning_gym" and meta.get("source_dataset"):
        return str(meta["source_dataset"])
    return task


def token_count(metadata):
    obj = metadata_obj(metadata)
    try:
        return int(obj.get("_prompt_tokens", 0)) + int(obj.get("_answer_tokens", 0))
    except Exception:
        return 0


def normalize_row(row):
    metadata = row.get("metadata", "{}")
    if isinstance(metadata, dict):
        metadata = json.dumps(metadata, ensure_ascii=False)
    elif metadata is None:
        metadata = "{}"
    else:
        metadata = str(metadata)
    meta = metadata_obj(metadata)
    cot = str(row.get("cot") or meta.get("cot") or "")
    return {
        "prompt": str(row.get("prompt", "")),
        "answer": str(row.get("answer", "")),
        "metadata": metadata,
        "task": metadata_task(row, meta),
        "cot": cot,
        "level": int(row.get("level", metadata_level(metadata)) or 0),
        "mode": str(row.get("mode") or "instruct"),
    }


def source_iter(args):
    cache_dir = os.path.join(args.run_dir, "huggingface", "datasets")
    if args.source == "staging":
        ds = load_dataset(
            "reasoning-core/staging",
            split="train",
            streaming=True,
            revision=args.source_revision,
            cache_dir=cache_dir,
        )
        iterator = ds
    elif os.path.isfile(args.source):
        def lines():
            with open(args.source, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        iterator = lines()
    elif os.path.isdir(args.source):
        files = sorted(
            os.path.join(args.source, name)
            for name in os.listdir(args.source)
            if name.endswith(".jsonl")
        )
        def many_lines():
            for path in files:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
        iterator = many_lines()
    else:
        raise FileNotFoundError(f"unsupported source: {args.source}")

    for row_i, row in enumerate(iterator):
        if args.smoke_test and row_i >= args.smoke_test:
            break
        yield row_i, normalize_row(row)


def source_total(args):
    if args.smoke_test:
        return args.smoke_test
    if args.source == "staging":
        try:
            builder = load_dataset_builder(
                "reasoning-core/staging",
                revision=args.source_revision,
                cache_dir=os.path.join(args.run_dir, "huggingface", "datasets"),
            )
            split = builder.info.splits.get("train") if builder.info.splits else None
            return None if split is None else split.num_examples
        except Exception:
            return None
    if os.path.isfile(args.source):
        return None
    return None


@dataclass
class Ewma:
    alpha: float = 0.2
    value: float | None = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


@dataclass
class Monitor:
    run_dir: str
    phase: str
    total: int | None = None
    status_every: float = 15.0
    metrics_every: float = 30.0
    started: float = field(default_factory=time.time)
    last_tick: float = field(default_factory=time.time)
    last_metrics: float = field(default_factory=time.time)
    last_rows: int = 0
    rate: Ewma = field(default_factory=Ewma)
    extra: dict = field(default_factory=dict)

    def tick(self, rows, force=False, **extra):
        self.extra.update(extra)
        now = time.time()
        dt = max(now - self.last_tick, 1e-9)
        if force or now - self.last_tick >= self.status_every:
            recent = (rows - self.last_rows) / dt
            rate = self.rate.update(recent)
            eta = None
            if self.total and rate and rate > 0:
                eta = max(0, (self.total - rows) / rate)
            payload = {
                "time": now_iso(),
                "phase": self.phase,
                "rows": rows,
                "total": self.total,
                "pct": None if not self.total else round(rows / self.total * 100, 2),
                "rows_per_second": round(rate or 0, 2),
                "eta_seconds": None if eta is None else int(eta),
                **self.extra,
            }
            self.write_status(payload)
            if now - self.last_metrics >= self.metrics_every or force:
                with open(os.path.join(self.run_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
                self.last_metrics = now
            msg = f"[{self.phase}] {rows:,}"
            if self.total:
                msg += f" / {self.total:,} ({rows / self.total * 100:.1f}%)"
            msg += f" {rate or 0:,.0f} rows/s"
            if eta is not None:
                msg += f" ETA {eta / 60:.1f}m"
            if extra:
                msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())
            print(msg, flush=True)
            self.last_tick = now
            self.last_rows = rows

    def write_status(self, payload):
        path = os.path.join(self.run_dir, "status.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)


class AnswerPools:
    def __init__(self, cap=8192, seed=42):
        self.cap = cap
        self.seed = seed
        self.heaps = defaultdict(list)
        self.selected = defaultdict(set)

    def add(self, task, answer):
        if self.cap <= 0:
            self.selected[task].add(answer)
            return
        chosen = self.selected[task]
        if answer in chosen:
            return
        score = xxhash.xxh64(str(answer).encode(), seed=self.seed).intdigest()
        heap = self.heaps[task]
        item = (-score, answer)
        if len(heap) < self.cap:
            import heapq
            heapq.heappush(heap, item)
            chosen.add(answer)
        elif score < -heap[0][0]:
            import heapq
            _, dropped = heapq.heapreplace(heap, item)
            chosen.remove(dropped)
            chosen.add(answer)

    def as_dict(self):
        if self.cap <= 0:
            return {task: list(values) for task, values in self.selected.items()}
        return {task: [answer for _, answer in heap] for task, heap in self.heaps.items()}


def init_run_dir(args):
    os.makedirs(args.work_root, exist_ok=True)
    return tempfile.mkdtemp(prefix="stream-run-", dir=args.work_root)


def isolate_runtime_dirs(run_dir, respect_env=False):
    if respect_env:
        return
    tmp_dir = os.path.join(run_dir, "tmp")
    hf_home = os.path.join(run_dir, "huggingface")
    hf_datasets = os.path.join(hf_home, "datasets")
    hf_hub = os.path.join(hf_home, "hub")
    hf_assets = os.path.join(hf_home, "assets")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = hf_datasets
    os.environ["HF_HUB_CACHE"] = hf_hub
    os.environ["HF_ASSETS_CACHE"] = hf_assets
    os.environ["XDG_CACHE_HOME"] = os.path.join(run_dir, "xdg_cache")
    tempfile.tempdir = tmp_dir
    try:
        import datasets.config as datasets_config
        datasets_config.HF_CACHE_HOME = hf_home
        datasets_config.HF_DATASETS_CACHE = hf_datasets
        datasets_config.HF_MODULES_CACHE = os.path.join(hf_home, "modules")
    except Exception:
        pass
    try:
        import huggingface_hub.constants as hub_constants
        hub_constants.HF_HOME = hf_home
        hub_constants.HF_HUB_CACHE = hf_hub
        hub_constants.HUGGINGFACE_HUB_CACHE = hf_hub
        hub_constants.HF_ASSETS_CACHE = hf_assets
    except Exception:
        pass


def preflight_hub_write(args):
    """Fail before preprocessing when Hub authentication cannot create the target."""
    if args.dry_run:
        return
    repo_id = f"reasoning-core/{args.dataset_name}"
    api = HfApi()
    try:
        identity = api.whoami()
        api.create_repo(
            repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Hub write preflight failed for {repo_id}; set a valid HF_TOKEN "
            "with write access before starting preprocessing"
        ) from exc
    print(f"hub write preflight ok: {identity.get('name', 'unknown')} -> {repo_id}", flush=True)


def write_manifest(run_dir, **data):
    path = os.path.join(run_dir, "manifest.json")
    current = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            current = json.load(f)
    current.update(data)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def open_bucket_files(bucket_dir, n_buckets):
    os.makedirs(bucket_dir, exist_ok=True)
    files = []
    for i in range(n_buckets):
        files.append(open(os.path.join(bucket_dir, f"bucket-{i:04d}.bin"), "ab", buffering=1024 * 1024))
    return files


def pass1_partition(args, run_dir, total):
    bucket_dir = os.path.join(run_dir, "dedup")
    files = open_bucket_files(bucket_dir, args.dedup_buckets)
    task_counts = Counter()
    task_to_id = {}
    row_task_ids = array("I")
    token_sum = 0
    answer_pools = AnswerPools(args.answer_pool_max_per_task, args.seed)
    rows_seen = 0
    accepted_pre_dedup = 0
    monitor = Monitor(run_dir, "pass1_partition", total=total, status_every=args.status_every)
    try:
        for row_i, row in source_iter(args):
            rows_seen = row_i + 1
            task = row["task"]
            task_id = task_to_id.setdefault(task, len(task_to_id))
            row_task_ids.append(task_id)
            prompt = row["prompt"]
            if len(prompt) >= args.max_prompt_chars:
                if rows_seen % args.tick_rows == 0:
                    monitor.tick(rows_seen, accepted=accepted_pre_dedup)
                continue
            lo, hi = deduplication_hash(row)
            bucket = lo & (args.dedup_buckets - 1) if args.dedup_buckets & (args.dedup_buckets - 1) == 0 else lo % args.dedup_buckets
            files[bucket].write(RECORD.pack(lo, hi, row_i))
            accepted_pre_dedup += 1
            task_counts[task] += 1
            token_sum += token_count(row["metadata"])
            answer_pools.add(task, row["answer"])
            if rows_seen % args.tick_rows == 0:
                monitor.tick(rows_seen, accepted=accepted_pre_dedup)
    finally:
        for f in files:
            f.close()

    stats = {
        "rows_seen": rows_seen,
        "accepted_pre_dedup": accepted_pre_dedup,
        "task_counts_pre_dedup": dict(task_counts),
        "task_to_id": task_to_id,
        "token_sum_pre_dedup": token_sum,
        "answer_pools": answer_pools.as_dict(),
    }
    with open(os.path.join(run_dir, "pass1_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f)
    with open(os.path.join(run_dir, "row_task_ids.u32"), "wb") as f:
        row_task_ids.tofile(f)
    monitor.tick(rows_seen, force=True, accepted=accepted_pre_dedup)
    write_manifest(run_dir, pass1_partition_done=True, rows_seen=rows_seen)
    return stats


def set_bitmap_bit(bitmap, idx):
    bitmap[idx >> 3] |= 1 << (idx & 7)


def get_bitmap_bit(bitmap, idx):
    return bool(bitmap[idx >> 3] & (1 << (idx & 7)))


def pass1_resolve(args, run_dir, rows_seen, task_to_id):
    bucket_dir = os.path.join(run_dir, "dedup")
    bitmap = bytearray((rows_seen + 7) // 8)
    bitmap_array = np.frombuffer(bitmap, dtype=np.uint8)
    row_task_ids = np.fromfile(os.path.join(run_dir, "row_task_ids.u32"), dtype=np.uint32)
    id_to_task = {v: k for k, v in task_to_id.items()}
    unique_counts = Counter()
    unique = 0
    monitor = Monitor(run_dir, "pass1_resolve", total=args.dedup_buckets, status_every=args.status_every)
    dtype = np.dtype([("lo", "<u8"), ("hi", "<u8"), ("idx", "<u8")])
    for bucket_i in range(args.dedup_buckets):
        path = os.path.join(bucket_dir, f"bucket-{bucket_i:04d}.bin")
        if not os.path.exists(path):
            monitor.tick(bucket_i + 1, unique=unique)
            continue
        size = os.path.getsize(path)
        if size:
            arr = np.fromfile(path, dtype=dtype)
            if len(arr):
                arr.sort(order=["lo", "hi", "idx"])
                first = np.empty(len(arr), dtype=bool)
                first[0] = True
                if len(arr) > 1:
                    first[1:] = (arr["lo"][1:] != arr["lo"][:-1]) | (arr["hi"][1:] != arr["hi"][:-1])
                indices = arr["idx"][first].astype(np.int64, copy=False)
                byte_indices = indices >> 3
                bit_masks = (1 << (indices & 7)).astype(np.uint8)
                np.bitwise_or.at(bitmap_array, byte_indices, bit_masks)
                task_ids = row_task_ids[indices]
                counts = np.bincount(task_ids, minlength=len(id_to_task))
                for task_id in np.flatnonzero(counts):
                    unique_counts[id_to_task[int(task_id)]] += int(counts[task_id])
                unique += int(len(indices))
                del arr
                gc.collect()
        os.remove(path)
        write_manifest(run_dir, dedup_bucket_done=bucket_i, unique_rows=unique)
        monitor.tick(bucket_i + 1, unique=unique)
    bitmap_path = os.path.join(bucket_dir, "keep.bitmap")
    with open(bitmap_path, "wb") as f:
        f.write(bitmap)
    monitor.tick(args.dedup_buckets, force=True, unique=unique)
    with open(os.path.join(run_dir, "unique_task_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(unique_counts), f)
    write_manifest(run_dir, pass1_resolve_done=True, unique_rows=unique, bitmap_path=bitmap_path)
    del row_task_ids
    return bitmap, unique, unique_counts


def apply_task_cap(counts, top_k):
    if not counts or top_k <= 0:
        return dict(counts)
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_counts) <= top_k:
        return dict(counts)
    limit = int(sorted_counts[top_k][1])
    return {task: min(count, limit) for task, count in counts.items()}


def compute_task_targets(unique_counts, dataset_name):
    first = apply_task_cap(unique_counts, top_k=max(1, len(unique_counts) - 6))
    if dataset_name == "reasoning-gym":
        return first
    return apply_task_cap(first, top_k=len(first) // 2)


def safe_score(cand, row, timeout=1.0):
    global _SCORE_DEVNULL

    def handler(signum, frame):
        raise Timeout()
    if _SCORE_DEVNULL is None or _SCORE_DEVNULL.closed:
        _SCORE_DEVNULL = open(os.devnull, "w")
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    t0 = time.time()
    try:
        with contextlib.redirect_stderr(_SCORE_DEVNULL):
            return score_answer(cand, edict(row)), time.time() - t0
    except Exception:
        return None, time.time() - t0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def verification_row(row, pools, args):
    metrics = Counter({"verification_rows": 1})
    ans = row["answer"]
    task = row["task"]
    chosen, label = ans, "Yes"
    pool = pools.get(task, [])
    if pool and stable_u01(row["prompt"], "verification_negative", seed=args.seed) < 0.5:
        rng = random.Random(xxhash.xxh64(row["prompt"].encode(), seed=args.seed).intdigest())
        for cand in rng.sample(pool, min(args.verif_max_tries, len(pool))):
            if cand == ans:
                continue
            if args.verif_negative_strategy == "same_task":
                chosen, label = cand, "No"
                metrics["verification_negatives"] += 1
                break
            metrics["score_attempts"] += 1
            score, elapsed = safe_score(cand, row, timeout=args.score_timeout)
            metrics["score_seconds"] += elapsed
            if score is not None and score != 1:
                chosen, label = cand, "No"
                metrics["score_successes"] += 1
                metrics["verification_negatives"] += 1
                break
            if score is None:
                metrics["score_unknown"] += 1
    out = dict(row)
    out["prompt"] = f"{row['prompt']}\nAnswer:\n{chosen}\nCorrect? (Yes/No)"
    out["answer"] = label
    out["mode"] = "verification"
    out["cot"] = ""
    return out, metrics


def init_verify_worker(pools, args_dict):
    global _VERIFY_POOLS, _VERIFY_ARGS
    _VERIFY_POOLS = pools
    _VERIFY_ARGS = argparse.Namespace(**args_dict)
    random.seed(os.getpid() ^ int(time.time() * 1e6))


def verify_worker(item):
    row_i, row = item
    out, metrics = verification_row(row, _VERIFY_POOLS, _VERIFY_ARGS)
    return row_i, out, dict(metrics)


def maybe_few_shot(row, shot_pools, args):
    if args.few_shot_ratio <= 0 or row["mode"] != "instruct":
        return row
    if stable_u01(row["prompt"], "few_shot", seed=args.seed) >= args.few_shot_ratio:
        return row
    pool = shot_pools.get(row["task"], [])
    if not pool:
        return row
    rng = random.Random(xxhash.xxh64(row["prompt"].encode(), seed=args.seed + 1).intdigest())
    shot = rng.choice(pool)
    if shot["prompt"] == row["prompt"]:
        return row
    out = dict(row)
    out["prompt"] = f"{shot['prompt']}\nAnswer:\n{shot['answer']}\n\n{row['prompt']}\nAnswer:\n"
    out["mode"] = "few_shot"
    return out


def maybe_cot(row, args):
    if args.cot_ratio <= 0 or row["mode"] == "verification":
        return row
    cot = row.get("cot") or metadata_obj(row["metadata"]).get("cot")
    if not cot or stable_u01(row["prompt"], "cot", seed=args.seed) >= args.cot_ratio:
        return row
    out = dict(row)
    out["prompt"] = f"/trace {row['prompt']}"
    out["answer"] = f"<trace>\n{cot}\n</trace>\n{row['answer']}"
    out["mode"] = "cot"
    return out


class ShardWriter:
    def __init__(self, args, run_dir):
        self.args = args
        self.run_dir = run_dir
        self.output_dir = os.path.join(run_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.api = HfApi()
        self.repo_id = f"reasoning-core/{args.dataset_name}"
        self.schema = output_schema(args.dataset_name)
        self.upload_prefix = args.upload_prefix.strip("/")
        self.rows = {"train": [], "test": []}
        self.shard_i = {"train": 0, "test": 0}
        self.uploaded_rows = 0
        self.uploaded_shards = []
        if not args.dry_run:
            if args.clear_existing_data:
                self.clear_existing_data()

    def clear_existing_data(self):
        files = self.api.list_repo_files(self.repo_id, repo_type="dataset")
        deletes = [
            CommitOperationDelete(path_in_repo=path)
            for path in files
            if path.endswith(".parquet")
        ]
        if deletes:
            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=deletes,
                commit_message="clear previous streaming rebuild shards",
            )

    def add(self, row):
        split = row.pop("_split", "train")
        self.rows[split].append({k: row.get(k, "" if k != "level" else 0) for k in self.schema.names})
        if len(self.rows[split]) >= self.args.shard_rows:
            self.flush(split)

    def flush(self, split=None):
        splits = [split] if split else ["train", "test"]
        for split_name in splits:
            self._flush_split(split_name)

    def _flush_split(self, split):
        if not self.rows[split]:
            return
        shard_i = self.shard_i[split]
        shard_name = f"{self.upload_prefix}/{split}-{shard_i:05d}.parquet"
        path = os.path.join(self.output_dir, f"{split}-{shard_i:05d}.parquet")
        table = pa.Table.from_pylist(self.rows[split], schema=self.schema)
        pq.write_table(table, path, compression="zstd")
        row_count = len(self.rows[split])
        sha = sha256_file(path)
        if not self.args.dry_run:
            self.api.upload_file(
                path_or_fileobj=path,
                path_in_repo=shard_name,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"upload {split} shard {shard_i:05d}",
            )
        if not self.args.keep_output_shards:
            os.remove(path)
        self.uploaded_rows += row_count
        self.uploaded_shards.append({"path": shard_name, "rows": row_count, "sha256": sha})
        write_manifest(
            self.run_dir,
            next_shard={**self.shard_i, split: self.shard_i[split] + 1},
            uploaded_rows=self.uploaded_rows,
            uploaded_shards=self.uploaded_shards,
        )
        self.rows[split].clear()
        self.shard_i[split] += 1
        del table
        gc.collect()


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def process_transform_batch(batch, pool, answer_pools, shot_pools, writer, args):
    if not batch:
        return 0, Counter()

    metrics = Counter()
    out_by_row = {}
    verify_items = [(row_i, row) for row_i, row, is_verif in batch if is_verif]

    if verify_items:
        if pool is None:
            verified = []
            for row_i, row in verify_items:
                out, item_metrics = verification_row(row, answer_pools, args)
                verified.append((row_i, out, dict(item_metrics)))
        else:
            verified = pool.map(verify_worker, verify_items, chunksize=args.verif_chunksize)
        for row_i, out, item_metrics in verified:
            out_by_row[row_i] = out
            metrics.update(item_metrics)

    rows_out = 0
    for row_i, row, is_verif in batch:
        out = out_by_row[row_i] if is_verif else dict(row)
        out = maybe_cot(out, args)
        out = maybe_few_shot(out, shot_pools, args)
        out["_split"] = "test" if stable_u01(row_i, row["prompt"], "test_split", seed=args.seed) < args.test_ratio else "train"
        if out["mode"] == "instruct" and len(shot_pools[out["task"]]) < args.few_shot_pool_per_task:
            shot_pools[out["task"]].append({"prompt": out["prompt"], "answer": out["answer"]})
        writer.add(out)
        rows_out += 1
    return rows_out, metrics


def pass2_transform(args, run_dir, bitmap, unique_counts, answer_pools, total):
    targets = compute_task_targets(unique_counts, args.dataset_name)
    remaining = dict(unique_counts)
    remaining_keep = dict(targets)
    shot_pools = defaultdict(list)
    writer = ShardWriter(args, run_dir)
    monitor = Monitor(run_dir, "pass2_transform", total=total, status_every=args.status_every)
    rows_out = 0
    rows_kept = 0
    rows_seen = 0
    task_seen = Counter()
    score_metrics = Counter()
    batch = []
    pool = None

    if args.verif_workers > 1:
        started = time.time()
        for task in sorted(answer_pools):
            getattr(get_task(task), "score_answer")
        print(f"preloaded {len(answer_pools)} scorers in {time.time() - started:.1f}s", flush=True)
        worker_args = vars(args).copy()
        pool = mp.Pool(
            processes=args.verif_workers,
            initializer=init_verify_worker,
            initargs=(answer_pools, worker_args),
            maxtasksperchild=args.verif_worker_maxtasks,
        )

    try:
        for row_i, row in source_iter(args):
            rows_seen = row_i + 1
            if row_i >= len(bitmap) * 8 or not get_bitmap_bit(bitmap, row_i):
                if rows_seen % args.tick_rows == 0:
                    monitor.tick(rows_seen, kept=rows_kept, output=rows_out, shard=writer.shard_i, **score_metrics)
                continue
            if len(row["prompt"]) >= args.max_prompt_chars:
                if rows_seen % args.tick_rows == 0:
                    monitor.tick(rows_seen, kept=rows_kept, output=rows_out, shard=writer.shard_i, **score_metrics)
                continue

            task = row["task"]
            rem = remaining.get(task, 0)
            keep = remaining_keep.get(task, 0)
            take = keep > 0 and (keep >= rem or stable_u01(row_i, task, "balance", seed=args.seed) < keep / max(rem, 1))
            remaining[task] = max(0, rem - 1)
            if take:
                remaining_keep[task] = keep - 1
                rows_kept += 1
                task_seen[task] += 1
                is_verif = stable_u01(row_i, row["prompt"], "verif_split", seed=args.seed) < args.verif_ratio
                batch.append((row_i, row, is_verif))
                if len(batch) >= args.transform_batch_rows:
                    n_out, batch_metrics = process_transform_batch(batch, pool, answer_pools, shot_pools, writer, args)
                    rows_out += n_out
                    score_metrics.update(batch_metrics)
                    batch.clear()

            if rows_seen % args.tick_rows == 0:
                monitor.tick(rows_seen, kept=rows_kept, output=rows_out, shard=writer.shard_i, **score_metrics)

        if batch:
            n_out, batch_metrics = process_transform_batch(batch, pool, answer_pools, shot_pools, writer, args)
            rows_out += n_out
            score_metrics.update(batch_metrics)
            batch.clear()
        writer.flush()
        monitor.tick(total or rows_seen, force=True, kept=rows_kept, output=rows_out, shard=writer.shard_i, **score_metrics)
        write_manifest(run_dir, pass2_done=True, output_rows=rows_out, kept_rows=rows_kept, score_metrics=dict(score_metrics))
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def notify(subject, body=""):
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        print("email skipped: RESEND_API_KEY is not set")
        return
    try:
        import resend
        resend.api_key = api_key
        resend.Emails.send({"from": "onboarding@resend.dev", "to": "damien.sileo@gmail.com",
                            "subject": subject, "html": f"<p>{html.escape(body)}</p>"})
    except Exception as e:
        print(f"email failed: {e}")


def close_score_devnull():
    global _SCORE_DEVNULL
    if _SCORE_DEVNULL is not None:
        try:
            _SCORE_DEVNULL.close()
        except Exception:
            pass
        _SCORE_DEVNULL = None


def copy_run_logs(work_root, run_dir, success):
    log_dir = os.path.join(work_root, "logs", os.path.basename(run_dir))
    os.makedirs(log_dir, exist_ok=True)
    for name in (
        "manifest.json",
        "metrics.jsonl",
        "status.json",
        "unique_task_counts.json",
    ):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(log_dir, name))
    with open(os.path.join(log_dir, "final.json"), "w", encoding="utf-8") as f:
        json.dump({"success": success, "run_dir": run_dir, "finished": now_iso()}, f, indent=2)
        f.write("\n")
    return log_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="staging")
    ap.add_argument("--source_revision", default=None)
    ap.add_argument("--dataset_name", default="procedural-pile")
    ap.add_argument("--work_root", default=os.path.join(os.environ.get("TMPDIR", os.path.expanduser("~/tmp")), "rc_streaming"))
    ap.add_argument("--resume_run_dir",
                    help="Resume pass 2 from a run with completed pass-1 artifacts")
    ap.add_argument("--upload_prefix", default=None,
                    help="Repo prefix for parquet shards; defaults to data")
    ap.add_argument("--keep_work_dir", action="store_true")
    ap.add_argument("--keep_output_shards", action="store_true",
                    help="Keep local parquet shards after write/upload for debugging")
    ap.add_argument("--clear_existing_data", action="store_true",
                    help="Delete existing data/*.parquet files in the target dataset before upload")
    ap.add_argument("--respect_env_cache", action="store_true",
                    help="Do not redirect HF/TMP caches under the run directory")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--dedup_buckets", type=int, default=512)
    ap.add_argument("--max_prompt_chars", type=int, default=50_000)
    ap.add_argument("--answer_pool_max_per_task", type=int, default=8192)
    ap.add_argument("--few_shot_pool_per_task", type=int, default=512)
    ap.add_argument("--few_shot_ratio", type=float, default=0.07)
    ap.add_argument("--cot_ratio", type=float, default=0.0)
    ap.add_argument("--verif_ratio", type=float, default=0.125)
    ap.add_argument("--test_ratio", type=float, default=0.01)
    ap.add_argument("--verif_max_tries", type=int, default=2)
    ap.add_argument("--verif_workers", type=int, default=6)
    ap.add_argument("--verif_chunksize", type=int, default=32)
    ap.add_argument("--verif_worker_maxtasks", type=int, default=1000)
    ap.add_argument("--verif_negative_strategy", choices=["same_task", "scored_same_task"], default="scored_same_task")
    ap.add_argument("--score_timeout", type=float, default=1.0)
    ap.add_argument("--transform_batch_rows", type=int, default=2048)
    ap.add_argument("--shard_rows", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--status_every", type=float, default=15.0)
    ap.add_argument("--tick_rows", type=int, default=8192,
                    help="Only run monitor bookkeeping every N input rows")
    args = ap.parse_args()

    if args.dedup_buckets <= 0:
        raise ValueError("--dedup_buckets must be positive")
    if args.tick_rows <= 0:
        raise ValueError("--tick_rows must be positive")
    if args.shard_rows <= 0:
        raise ValueError("--shard_rows must be positive")
    if args.transform_batch_rows <= 0:
        raise ValueError("--transform_batch_rows must be positive")
    if args.verif_workers < 1:
        raise ValueError("--verif_workers must be >= 1")
    if args.verif_chunksize <= 0:
        raise ValueError("--verif_chunksize must be positive")
    if not 0 <= args.verif_ratio <= 1:
        raise ValueError("--verif_ratio must be between 0 and 1")
    if not 0 <= args.test_ratio <= 1:
        raise ValueError("--test_ratio must be between 0 and 1")
    if not 0 <= args.few_shot_ratio <= 1:
        raise ValueError("--few_shot_ratio must be between 0 and 1")
    if not 0 <= args.cot_ratio <= 1:
        raise ValueError("--cot_ratio must be between 0 and 1")

    if args.resume_run_dir:
        run_dir = os.path.abspath(args.resume_run_dir)
        with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
            resume_manifest = json.load(f)
        if not resume_manifest.get("pass1_resolve_done"):
            raise RuntimeError(f"pass 1 is incomplete in {run_dir}")
        if resume_manifest.get("deduplication_version") != DEDUPLICATION_VERSION:
            raise RuntimeError(
                f"pass 1 in {run_dir} used an older deduplication scheme; restart preprocessing"
            )
        args.source = resume_manifest["source"]
        args.source_revision = resume_manifest["source_revision"]
        args.dataset_name = resume_manifest["dataset_name"]
        if args.upload_prefix is None:
            args.upload_prefix = resume_manifest["upload_prefix"]
    else:
        run_dir = init_run_dir(args)
    args.run_dir = run_dir
    if args.upload_prefix is None:
        args.upload_prefix = "data"
    isolate_runtime_dirs(run_dir, respect_env=args.respect_env_cache)
    preflight_hub_write(args)
    if args.source == "staging" and args.source_revision is None:
        args.source_revision = HfApi().repo_info("reasoning-core/staging", repo_type="dataset").sha
        print(f"pinned staging revision: {args.source_revision}", flush=True)
    print(f"run dir: {run_dir}", flush=True)
    if args.resume_run_dir:
        write_manifest(run_dir, resumed=now_iso(), upload_prefix=args.upload_prefix)
    else:
        write_manifest(
            run_dir,
            started=now_iso(),
            source=args.source,
            source_revision=args.source_revision,
            dataset_name=args.dataset_name,
            upload_prefix=args.upload_prefix,
            dry_run=args.dry_run,
            smoke_test=args.smoke_test,
            deduplication_version=DEDUPLICATION_VERSION,
        )
    total = source_total(args)
    stats = None
    bitmap = None
    success = False
    try:
        if args.resume_run_dir:
            with open(os.path.join(run_dir, "pass1_stats.json"), encoding="utf-8") as f:
                stats = json.load(f)
            with open(os.path.join(run_dir, "dedup", "keep.bitmap"), "rb") as f:
                bitmap = bytearray(f.read())
            with open(os.path.join(run_dir, "unique_task_counts.json"), encoding="utf-8") as f:
                unique_counts = json.load(f)
            unique = sum(unique_counts.values())
            print(f"resuming pass 2 from {unique:,} unique rows", flush=True)
        else:
            stats = pass1_partition(args, run_dir, total)
            bitmap, unique, unique_counts = pass1_resolve(args, run_dir, stats["rows_seen"], stats["task_to_id"])
        missing = set(list_tasks()) - set(unique_counts)
        if args.source == "staging" and missing:
            print(f"missing tasks: {missing}", flush=True)
        pass2_transform(args, run_dir, bitmap, unique_counts, stats["answer_pools"], total or stats["rows_seen"])
        notify("RC streaming preprocess done", f"run dir: {run_dir}")
        success = True
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        notify("RC streaming preprocess failed", tb[-8000:])
        raise
    finally:
        close_score_devnull()
        try:
            log_dir = copy_run_logs(args.work_root, run_dir, success)
            print(f"copied run logs: {log_dir}", flush=True)
        except Exception as exc:
            print(f"WARN failed to copy run logs: {exc}", file=sys.stderr, flush=True)
        if not success or args.keep_work_dir:
            print(f"kept work dir: {run_dir}", flush=True)
        else:
            try:
                shutil.rmtree(run_dir)
            except Exception as exc:
                print(f"WARN failed to clean work dir {run_dir}: {exc}", file=sys.stderr, flush=True)
            else:
                print(f"cleaned work dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
