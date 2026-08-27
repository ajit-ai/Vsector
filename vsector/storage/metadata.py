"""MetadataStore abstraction. In prod PostgreSQL, here in-memory + JSON persistence."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict

from ..models.namespace import Namespace


class MetadataStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._data: Dict[str, Namespace] = {}
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                for k, v in raw.items():
                    self._data[k] = Namespace.model_validate(v)
            except Exception:
                pass

    def _persist(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {k: v.model_dump(mode="json") for k, v in self._data.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def create(self, ns: Namespace) -> Namespace:
        with self._lock:
            if ns.name in self._data:
                raise ValueError(f"namespace {ns.name!r} already exists")
            self._data[ns.name] = ns
        self._persist()
        return ns

    def get(self, name: str) -> Namespace | None:
        with self._lock:
            return self._data.get(name)

    def delete(self, name: str) -> None:
        with self._lock:
            if name not in self._data:
                raise KeyError(name)
            del self._data[name]
        self._persist()

    def list(self) -> list[Namespace]:
        with self._lock:
            return list(self._data.values())

    def upsert(self, ns: Namespace) -> Namespace:
        with self._lock:
            self._data[ns.name] = ns
        self._persist()
        return ns
