"""MetadataStore: pluggable Postgres + JSON fallback. In prod PostgreSQL via asyncpg/psycopg, here JSON for single-node."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ..models.namespace import Namespace

logger = logging.getLogger(__name__)


class _JsonBackend:
    def __init__(self, path: Path | None):
        self.path = path
        self._data: dict[str, Namespace] = {}
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                for k, v in raw.items():
                    self._data[k] = Namespace.model_validate(v)
            except Exception as e:
                logger.warning(f"metadata json load failed: {e}")

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


class _PostgresBackend:
    """Postgres backend — uses DATABASE_URL / VSECTOR_DATABASE_URL. Falls back to Json if unavailable."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._pool = None
        self._available = False
        try:
            import psycopg  # type: ignore  # psycopg3
            self._psycopg = psycopg
            self._available = True
        except Exception:
            try:
                import psycopg2  # type: ignore

                self._psycopg = psycopg2  # type: ignore
                self._available = True
            except Exception as e:
                logger.warning(f"postgres driver not installed, using JSON fallback: {e}")
                self._available = False

    def _ensure_table(self, conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vsector_namespaces (
                name TEXT PRIMARY KEY,
                config JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        conn.commit()

    def create(self, ns: Namespace) -> Namespace:
        import psycopg  # type: ignore

        with psycopg.connect(self.dsn) as conn:  # type: ignore
            self._ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM vsector_namespaces WHERE name=%s", (ns.name,))
                if cur.fetchone():
                    raise ValueError(f"namespace {ns.name!r} already exists")
                cur.execute("INSERT INTO vsector_namespaces (name, config) VALUES (%s, %s)", (ns.name, json.dumps(ns.model_dump(mode="json"))))
                conn.commit()
        return ns

    def get(self, name: str) -> Namespace | None:
        import psycopg  # type: ignore

        with psycopg.connect(self.dsn) as conn:  # type: ignore
            with conn.cursor() as cur:
                cur.execute("SELECT config FROM vsector_namespaces WHERE name=%s", (name,))
                row = cur.fetchone()
                if not row:
                    return None
                return Namespace.model_validate(row[0] if isinstance(row[0], dict) else json.loads(row[0]))

    def delete(self, name: str) -> None:
        import psycopg  # type: ignore

        with psycopg.connect(self.dsn) as conn:  # type: ignore
            with conn.cursor() as cur:
                cur.execute("DELETE FROM vsector_namespaces WHERE name=%s", (name,))
                if cur.rowcount == 0:
                    raise KeyError(name)
                conn.commit()

    def list(self) -> list[Namespace]:
        import psycopg  # type: ignore

        with psycopg.connect(self.dsn) as conn:  # type: ignore
            with conn.cursor() as cur:
                cur.execute("SELECT config FROM vsector_namespaces")
                rows = cur.fetchall()
                return [Namespace.model_validate(r[0] if isinstance(r[0], dict) else json.loads(r[0])) for r in rows]

    def upsert(self, ns: Namespace) -> Namespace:
        import psycopg  # type: ignore

        with psycopg.connect(self.dsn) as conn:  # type: ignore
            self._ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO vsector_namespaces (name, config) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET config=EXCLUDED.config",
                    (ns.name, json.dumps(ns.model_dump(mode="json"))),
                )
                conn.commit()
        return ns


class MetadataStore:
    """Facade: Postgres if VSECTOR_DATABASE_URL / DATABASE_URL set and driver available, else JSON."""

    def __init__(self, path: str | Path | None = None, dsn: str | None = None):
        import os

        effective_dsn = dsn or os.getenv("VSECTOR_DATABASE_URL") or os.getenv("DATABASE_URL") or ""
        self._backend: _JsonBackend | _PostgresBackend
        self._is_postgres = False
        if effective_dsn:
            try:
                be = _PostgresBackend(effective_dsn)
                if be._available:
                    # probe
                    be.list()
                    self._backend = be
                    self._is_postgres = True
                    logger.info(f"MetadataStore: using Postgres {effective_dsn.split('@')[-1]}")
                else:
                    raise RuntimeError("no pg driver")
            except Exception as e:
                logger.warning(f"Postgres unavailable, falling back to JSON: {e}")
                self._backend = _JsonBackend(Path(path) if path else None)
        else:
            self._backend = _JsonBackend(Path(path) if path else None)

    def create(self, ns: Namespace) -> Namespace:
        return self._backend.create(ns)

    def get(self, name: str) -> Namespace | None:
        return self._backend.get(name)

    def delete(self, name: str) -> None:
        return self._backend.delete(name)

    def list(self) -> list[Namespace]:
        return self._backend.list()

    def upsert(self, ns: Namespace) -> Namespace:
        return self._backend.upsert(ns)

    @property
    def backend(self) -> str:
        return "postgres" if self._is_postgres else "json"
