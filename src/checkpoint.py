import json
import os
import threading
from typing import Iterable


def ensure_parent_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_jsonl(path: str) -> list:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


class JsonlWriter:
    def __init__(self, path: str, flush_every: int = 1):
        self.path = path
        self.flush_every = max(flush_every, 1)
        ensure_parent_dir(path)
        self._handle = open(path, "a", encoding="utf-8")
        self._written = 0
        self._lock = threading.Lock()

    def write(self, obj: dict):
        with self._lock:
            self._handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._written += 1
            if self._written % self.flush_every == 0:
                self._handle.flush()

    def flush(self):
        with self._lock:
            self._handle.flush()

    def close(self):
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def dedup_by_id(records: Iterable[dict], id_field: str = "_id", keep_id: bool = False) -> list:
    deduped = []
    seen = set()
    for record in records:
        rec = dict(record)
        record_id = rec.get(id_field)
        if record_id:
            if record_id in seen:
                continue
            seen.add(record_id)
        if not keep_id:
            rec.pop(id_field, None)
        deduped.append(rec)
    return deduped


class ProcessedTracker:
    def __init__(self, path: str):
        self.path = path
        self._seen = set()
        self._lock = threading.Lock()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fd:
                for line in fd:
                    line = line.strip()
                    if line:
                        self._seen.add(line)
        ensure_parent_dir(path)
        self._handle = open(path, "a", encoding="utf-8")

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._seen

    def add(self, key: str):
        with self._lock:
            if key in self._seen:
                return
            self._seen.add(key)
            self._handle.write(key + "\n")
            self._handle.flush()

    def close(self):
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
