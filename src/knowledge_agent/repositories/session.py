"""Shared Neo4j connection used by every repository.

Wraps ``langchain_neo4j.Neo4jGraph`` so that:

* Connection lifecycle (open / close) is centralized.
* Constructor reads credentials from ``Settings`` - the repositories never
  see a connection string.
* ``enhanced_schema=True`` is on by default; if the Aura instance lacks the
  APOC procedures it relies on, callers can construct with
  ``enhanced_schema=False`` from a higher layer.
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_neo4j import Neo4jGraph
from neo4j.exceptions import AuthError, ServiceUnavailable

from ..config import get_settings


class Neo4jConnectionError(RuntimeError):
    """Raised when the session can't be opened. Wraps the driver error with
    a one-line, user-actionable message instead of a 60-line traceback."""


class Neo4jSession:
    def __init__(self, *, enhanced_schema: bool = True) -> None:
        s = get_settings()
        try:
            self.graph = Neo4jGraph(
                url=s.neo4j_uri,
                username=s.neo4j_username,
                password=s.neo4j_password,
                database=s.neo4j_database,
                enhanced_schema=enhanced_schema,
                refresh_schema=False,
            )
        except Exception as e:
            # langchain_neo4j wraps the driver error as a ValueError but the
            # real cause (AuthError vs ServiceUnavailable vs ...) is chained
            # on ``__context__`` (implicit chaining; not ``__cause__``).
            root = _root_cause(e)
            if isinstance(root, AuthError):
                raise Neo4jConnectionError(
                    f"Neo4j REJECTED the credentials in .env.\n"
                    f"  URI:      {s.neo4j_uri}\n"
                    f"  Username: {s.neo4j_username!r}\n"
                    f"  Database: {s.neo4j_database!r}\n"
                    f"Copy NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / "
                    f"NEO4J_DATABASE verbatim from the credentials .txt Aura "
                    f"gave you on instance creation. Depending on the Aura "
                    f"edition the username and database may be 'neo4j' OR the "
                    f"instance id - trust the file, don't guess. If you've "
                    f"lost the file, reset the password from the Aura console. "
                    f"If the instance shows 'paused' (Aura Free pauses after 3 "
                    f"days idle), resume it.\nDriver said: {root}"
                ) from e
            if isinstance(root, ServiceUnavailable):
                raise Neo4jConnectionError(
                    f"Could not reach the Neo4j instance at {s.neo4j_uri!r}. "
                    f"Check the URI scheme (Aura needs 'neo4j+s://') and that "
                    f"the instance is running, not paused.\nDriver said: {root}"
                ) from e
            raise Neo4jConnectionError(
                f"Could not connect to Neo4j at {s.neo4j_uri!r} as user "
                f"{s.neo4j_username!r}. {type(root).__name__}: {root}"
            ) from e

    # ----- thin wrappers around Neo4jGraph ---------------------------------

    def query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.graph.query(cypher, params=params or {})

    def write_batch(self, cypher: str, rows: Iterable[dict[str, Any]]) -> None:
        """Cypher must reference ``$rows``."""
        self.graph.query(cypher, params={"rows": list(rows)})

    def refresh_schema(self) -> None:
        self.graph.refresh_schema()

    @property
    def schema(self) -> str:
        return self.graph.schema

    def close(self) -> None:
        driver = getattr(self.graph, "_driver", None)
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


def _root_cause(exc: BaseException) -> BaseException:
    """Walk both ``__cause__`` (explicit ``raise X from Y``) and
    ``__context__`` (implicit chaining from ``except: raise X``) to find the
    deepest error. langchain_neo4j uses the latter, so checking only
    ``__cause__`` misses the real ``AuthError``."""
    cur: BaseException = exc
    seen: set[int] = set()
    while True:
        nxt = cur.__cause__ or cur.__context__
        if nxt is None or id(nxt) in seen:
            return cur
        seen.add(id(nxt))
        cur = nxt
