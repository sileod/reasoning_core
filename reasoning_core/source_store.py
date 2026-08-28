"""Immutable, content-addressed storage for historical task sources."""

import hashlib
import os
from pathlib import Path
import tempfile


class SourceStore:
    """Store source snapshots by their SHA-256 digest."""

    def __init__(self, objects_dir=Path(".evolution") / "objects"):
        self.objects_dir = Path(objects_dir)

    def _path(self, source_id):
        if (not isinstance(source_id, str) or len(source_id) != 64
                or any(c not in "0123456789abcdef" for c in source_id)):
            raise ValueError("source_id must be a lowercase SHA-256 hex digest")
        return self.objects_dir / source_id[:2] / f"{source_id}.py"

    @staticmethod
    def _checked_source(path, source_id):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise KeyError(source_id) from None
        if hashlib.sha256(data).hexdigest() != source_id:
            raise RuntimeError(f"corrupt source object: {source_id}")
        return data

    def put(self, source):
        """Store *source* once and return its stable SHA-256 ID."""
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        data = source.encode("utf-8")
        source_id = hashlib.sha256(data).hexdigest()
        path = self._path(source_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            existing = self._checked_source(path, source_id)
        except KeyError:
            existing = None
        if existing is not None:
            return source_id

        fd, temporary = tempfile.mkstemp(prefix=f".{source_id}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # A hard link publishes the complete file without replacing an
                # object another writer may have created concurrently.
                os.link(temporary, path)
            except FileExistsError:
                self._checked_source(path, source_id)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return source_id

    def get(self, source_id):
        """Return the exact source associated with *source_id*."""
        return self._checked_source(self._path(source_id), source_id).decode("utf-8")


def snapshot_parent(parent_path, *, idea, hypothesis, changes, store=None):
    """Read and snapshot a parent, returning the exact source and child metadata."""
    source = Path(parent_path).read_bytes().decode("utf-8")
    parent_source_id = (store or SourceStore()).put(source)
    metadata = {
        "parent_source_id": parent_source_id,
        "idea": idea,
        "hypothesis": hypothesis,
        "changes": changes,
    }
    return source, metadata
