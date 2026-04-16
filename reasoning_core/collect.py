
import argparse, hashlib, json, os, tempfile, time
from pathlib import Path

home = Path.home()
tmp = home / "tmp"; tmp.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_DATASETS_CACHE=str(home / ".cache/huggingface"),
                  TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
tempfile.tempdir = str(tmp)

from datasets import disable_caching, load_dataset, Dataset
from huggingface_hub import HfApi, login
from nfsdict import NfsDict
from tqdm import tqdm

import random
from io import BytesIO
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq

disable_caching()


MAX_RETRIES = 5
RETRY_BASE = 70

ap = argparse.ArgumentParser()
ap.add_argument("--rc_path", default=os.getcwd())
ap.add_argument("--dataset_name", default="staging")
ap.add_argument("--org", default="reasoning-core")
ap.add_argument("--batch", type=int, default=10_000)
ap.add_argument("--num_proc", type=int, default=24)
ap.add_argument("--version", default="*")
ap.add_argument("--delete", action=argparse.BooleanOptionalAction, default=True, help="Delete files after upload")
args = ap.parse_args()


def file_key(p: str) -> str:
    try:
        st = os.stat(p)
        return hashlib.sha1(f"{p}\0{st.st_size}\0{st.st_mtime_ns}".encode()).hexdigest()[:20]
    except OSError:
        return hashlib.sha1(p.encode()).hexdigest()[:20]


def validate_jsonl(path: str) -> tuple[bool, str | None]:
    n = 0
    try:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    json.loads(line)
        return True, None
    except Exception as e:
        return False, f"line {n or '?'}: {e}"


def process_row(ex: dict) -> dict:
    raw = ex.get("metadata", "{}")
    if isinstance(raw, str):
        try: obj = json.loads(raw)
        except Exception: obj = {}
    else:
        obj, raw = raw, json.dumps(raw, ensure_ascii=False)
    return {
        "metadata": raw,
        "level": obj.get("_level") if isinstance(obj, dict) else None,
        "mode": "instruct",
    }


def file_iterator(rc_path: str, version: str, state: NfsDict, delete: bool = False):
    """Single lazy pass over the glob. Yields paths not yet in state."""
    deleted_count = 0
    for p in Path(rc_path, "generated_data").glob(f"{version}/*.jsonl"):
        ap = os.path.abspath(p)
        k = file_key(ap)
        if k not in state:
            yield ap
        elif delete and isinstance(state[k], dict) and state[k].get("s") == "done":
            try:
                os.unlink(ap)
                del state[k]
                deleted_count += 1
                if deleted_count % 1000 == 0:
                    s = state.get("__stats__", {})
                    s["done"] = s.get("done", 0) + 1000
                    state["__stats__"] = s
            except OSError:
                pass
    if deleted_count % 1000 != 0:
        s = state.get("__stats__", {})
        s["done"] = s.get("done", 0) + (deleted_count % 1000)
        state["__stats__"] = s


def build_batch(file_iter, state: NfsDict, size: int, pbar: tqdm) -> tuple[list[str], int]:
    """Pull exactly `size` good files from the *shared* iterator."""
    batch, bad = [], 0
    for p in file_iter:
        pbar.update(1)
        k = file_key(p)
        if k in state:
            continue
        ok, err = validate_jsonl(p)
        if ok:
            batch.append(p)
            pbar.set_postfix(good=len(batch), bad=bad, refresh=False)
        else:
            state[k] = {"s": "bad", "reason": (err or "invalid")[:200]}
            bad += 1
            pbar.set_postfix(good=len(batch), bad=bad, refresh=False)
        if len(batch) >= size:
            break
    return batch, bad


def batch_id(paths: list[str]) -> str:
    return hashlib.sha1(
        "\n".join(sorted(file_key(p) for p in paths)).encode()
    ).hexdigest()[:12]


def upload_with_retry(api: HfApi, local: str, remote: str, repo: str, msg: str):
    for attempt in range(MAX_RETRIES):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id=repo, repo_type="dataset", commit_message=msg)
            return
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e):
                wait = RETRY_BASE * (2 ** attempt)
                print(f"  ⏳ Rate-limited, waiting {wait}s "
                      f"(attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Upload failed after {MAX_RETRIES} retries")


# V2

def _load_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            if len(ex.get("prompt", "")) >= 50_000: continue
            rows.append({**ex, **process_row(ex)})
    return rows

def upload_shard(files, bid, api, repo, num_proc):
    shard = f"data/shard-{bid}.parquet"
    if api.file_exists(repo_id=repo, filename=shard, repo_type="dataset"):
        print(f"  ↺ {shard} exists on hub"); return shard, None

    print(f"  ⏳ Loading & processing {len(files):,} files...")
    with Pool(num_proc) as pool:
        rows = [r for batch in pool.imap_unordered(_load_file, files, chunksize=8) for r in batch]
    random.shuffle(rows)

    buf = BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf, compression="snappy")
    buf.seek(0)
    print(f"  📦 {shard} — {len(rows):,} rows, {buf.getbuffer().nbytes/1024/1024:.1f} MB")
    print(f"  📤 Uploading...")
    upload_with_retry(api, buf, shard, repo,
                      f"shard {bid} ({len(files)} files, {len(rows):,} rows)")
    print(f"  ✅ Uploaded!")
    return shard, len(rows)

# / V2



def mark_batch_done(state: NfsDict, paths: list[str], shard: str, delete: bool = False):
    t = time.time()
    deleted_count = 0
    for p in tqdm(paths, desc="  💾 Marking done", unit="f", leave=False):
        k = file_key(p)
        if delete:
            try:
                os.unlink(p)
                if k in state:
                    del state[k]
                deleted_count += 1
                continue
            except OSError:
                pass
        state[k] = {"s": "done", "shard": shard, "t": t}
    if deleted_count > 0:
        s = state.get("__stats__", {})
        s["done"] = s.get("done", 0) + deleted_count
        state["__stats__"] = s


def stats(state: NfsDict) -> tuple[int, int]:
    done = bad = 0
    s_obj = state.get("__stats__", {})
    if isinstance(s_obj, dict):
        done += s_obj.get("done", 0)
        bad += s_obj.get("bad", 0)
    for k, v in state.items():
        if isinstance(k, str) and k.startswith("__"): continue
        if isinstance(v, dict):
            if v.get("s") == "done": done += 1
            elif v.get("s") == "bad": bad += 1
    return done, bad


def main(args):
    repo = f"{args.org}/{args.dataset_name}"
    state_dir = os.path.join(args.rc_path, "upload_state")

    print(f"🕷️  collect → {repo}")
    print(f"   rc_path  = {args.rc_path}")
    print(f"   version  = {args.version}")
    print(f"   batch    = {args.batch}")
    print(f"   num_proc = {args.num_proc}")
    print(f"   delete   = {args.delete}")
    print(f"   Loading state...", flush=True)

    state = NfsDict(
        name=f"collect_{args.dataset_name}",
        base_dir=state_dir,
        serializer="json",
        progress_bar=True,
    )

    d, b = stats(state)
    print(f"   history  = {d} done, {b} bad ({len(state)} tracked)\n", flush=True)

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)

    # ── Resume pending ────────────────────────────────────────────────
    pending_data = state.get("__pending__")
    if pending_data and isinstance(pending_data, dict):
        pid = pending_data.get("id")
        pending = [p for p in pending_data.get("paths", [])
                   if os.path.exists(p) and file_key(p) not in state]
        if pending:
            good, nb = [], 0
            for p in tqdm(pending, desc="🔄 Validating pending", unit="f", leave=False):
                ok, err = validate_jsonl(p)
                if ok:
                    good.append(p)
                else:
                    state[file_key(p)] = {"s": "bad", "reason": (err or "")[:200]}
                    nb += 1
            if good:
                print(f"🔄 Resuming {pid} ({len(good)} files)")
                bid = batch_id(good)
                shard, rows = upload_shard(good, bid, api, repo, args.num_proc)
                mark_batch_done(state, good, shard, delete=args.delete)
                d, b = stats(state)
                print(f"  ✓ {d} done, {b} bad" +
                      (f", {rows:,} rows" if rows else "") + "\n")
    if "__pending__" in state:
        del state["__pending__"]

    # ── Main loop: single glob pass, interleaved scan→process→push ───
    try:
        n = 0
        file_iter = file_iterator(args.rc_path, args.version, state, delete=args.delete)
        pbar = tqdm(desc="🔍 Scanning", unit="f")

        while True:
            batch_files, nb = build_batch(file_iter, state, args.batch, pbar)
            if not batch_files:
                break

            pbar.set_description("⏸️  Uploading")

            n += 1
            bid = batch_id(batch_files)
            state["__pending__"] = {"id": bid, "paths": batch_files, "t": time.time()}

            d, b = stats(state)
            print(f"\n📤 #{n} [{bid}] {len(batch_files):,} files  ({d} done, {b} bad)")

            shard, rows = upload_shard(batch_files, bid, api, repo, args.num_proc)
            mark_batch_done(state, batch_files, shard, delete=args.delete)
            if "__pending__" in state:
                del state["__pending__"]

            d, b = stats(state)
            print(f"  🏁 {d} done, {b} bad" +
                  (f", {rows:,} rows this batch" if rows else "") + "\n")

            pbar.set_description("🔍 Scanning")

        pbar.close()
        d, b = stats(state)
        print(f"🎉 Done — {d} uploaded, {b} bad")

    except KeyboardInterrupt:
        pbar.close()
        d, b = stats(state)
        print(f"\n⚠️  Interrupted — {d} done, {b} bad")
        raise SystemExit(130)


if __name__ == "__main__":
    main(args)