
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


def load_file_state(state: NfsDict) -> tuple[set[str], set[str]]:
    """Index both legacy per-file entries and compact batch manifests."""
    done, bad = set(), set()
    for key, value in state.items():
        if not isinstance(value, dict):
            continue
        status = value.get("s")
        target = done if status == "done" else bad if status == "bad" else None
        if target is None:
            continue
        files = value.get("files")
        if isinstance(files, list):
            target.update(files)
        elif not key.startswith("__") and not key.startswith("batch:"):
            target.add(key)  # legacy one-file-per-key state
    return done, bad


def save_manifest(state: NfsDict, status: str, keys: list[str], **extra):
    if not keys:
        return
    mid = hashlib.sha1("\n".join(sorted(keys)).encode()).hexdigest()[:12]
    state[f"batch:{status}:{mid}"] = {"s": status, "files": keys, **extra}


def file_iterator(rc_path: str, version: str, done: set[str], bad: set[str],
                  delete: bool = False):
    """Single lazy pass over the glob. Yields paths not yet in state."""
    for p in Path(rc_path, "generated_data").glob(f"{version}/*.jsonl"):
        ap = os.path.abspath(p)
        k = file_key(ap)
        if k not in done and k not in bad:
            yield ap
        elif delete and k in done:
            try:
                os.unlink(ap)
            except OSError:
                pass


def build_batch(file_iter, known: set[str], size: int, pbar: tqdm):
    """Pull exactly `size` good files from the *shared* iterator."""
    batch, bad = [], {}
    for p in file_iter:
        pbar.update(1)
        k = file_key(p)
        if k in known:
            continue
        ok, err = validate_jsonl(p)
        if ok:
            batch.append(p)
            pbar.set_postfix(good=len(batch), bad=len(bad), refresh=False)
        else:
            bad[k] = (err or "invalid")[:200]
            known.add(k)
            pbar.set_postfix(good=len(batch), bad=len(bad), refresh=False)
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



def mark_batch_done(state: NfsDict, paths: list[str], shard: str,
                    done: set[str], delete: bool = False):
    keys = [file_key(p) for p in paths]
    # Persist the whole uploaded batch atomically before deleting any source.
    save_manifest(state, "done", keys, shard=shard, t=time.time())
    done.update(keys)
    for p in tqdm(paths, desc="  💾 Marking done", unit="f", leave=False):
        if delete:
            try:
                os.unlink(p)
            except OSError:
                pass


def stats(state: NfsDict, done: set[str], bad: set[str]) -> tuple[int, int]:
    s_obj = state.get("__stats__", {})
    historical_done = s_obj.get("done", 0) if isinstance(s_obj, dict) else 0
    historical_bad = s_obj.get("bad", 0) if isinstance(s_obj, dict) else 0
    return historical_done + len(done), historical_bad + len(bad)


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
    done, bad = load_file_state(state)

    d, b = stats(state, done, bad)
    print(f"   history  = {d} done, {b} bad ({len(done) + len(bad)} tracked)\n", flush=True)

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)

    # ── Resume pending ────────────────────────────────────────────────
    pending_data = state.get("__pending__")
    if pending_data and isinstance(pending_data, dict):
        pid = pending_data.get("id")
        pending = [p for p in pending_data.get("paths", [])
                   if os.path.exists(p) and file_key(p) not in done
                   and file_key(p) not in bad]
        if pending:
            good, bad_entries = [], {}
            for p in tqdm(pending, desc="🔄 Validating pending", unit="f", leave=False):
                ok, err = validate_jsonl(p)
                if ok:
                    good.append(p)
                else:
                    bad_entries[file_key(p)] = (err or "invalid")[:200]
            save_manifest(state, "bad", list(bad_entries),
                          reasons=bad_entries, t=time.time())
            bad.update(bad_entries)
            if good:
                print(f"🔄 Resuming {pid} ({len(good)} files)")
                bid = batch_id(good)
                shard, rows = upload_shard(good, bid, api, repo, args.num_proc)
                mark_batch_done(state, good, shard, done, delete=args.delete)
                d, b = stats(state, done, bad)
                print(f"  ✓ {d} done, {b} bad" +
                      (f", {rows:,} rows" if rows else "") + "\n")
    if "__pending__" in state:
        del state["__pending__"]

    # ── Main loop: single glob pass, interleaved scan→process→push ───
    try:
        n = 0
        file_iter = file_iterator(args.rc_path, args.version, done, bad,
                                  delete=args.delete)
        pbar = tqdm(desc="🔍 Scanning", unit="f")

        while True:
            batch_files, bad_entries = build_batch(file_iter, done | bad,
                                                    args.batch, pbar)
            save_manifest(state, "bad", list(bad_entries),
                          reasons=bad_entries, t=time.time())
            bad.update(bad_entries)
            if not batch_files:
                break

            pbar.set_description("⏸️  Uploading")

            n += 1
            bid = batch_id(batch_files)
            state["__pending__"] = {"id": bid, "paths": batch_files, "t": time.time()}

            d, b = stats(state, done, bad)
            print(f"\n📤 #{n} [{bid}] {len(batch_files):,} files  ({d} done, {b} bad)")

            shard, rows = upload_shard(batch_files, bid, api, repo, args.num_proc)
            mark_batch_done(state, batch_files, shard, done, delete=args.delete)
            if "__pending__" in state:
                del state["__pending__"]

            d, b = stats(state, done, bad)
            print(f"  🏁 {d} done, {b} bad" +
                  (f", {rows:,} rows this batch" if rows else "") + "\n")

            pbar.set_description("🔍 Scanning")

        pbar.close()
        d, b = stats(state, done, bad)
        print(f"🎉 Done — {d} uploaded, {b} bad")

    except KeyboardInterrupt:
        pbar.close()
        d, b = stats(state, done, bad)
        print(f"\n⚠️  Interrupted — {d} done, {b} bad")
        raise SystemExit(130)


if __name__ == "__main__":
    main(args)
