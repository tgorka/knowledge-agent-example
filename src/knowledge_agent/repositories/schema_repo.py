"""Schema-side operations: introspection, constraints, audit log.

Three responsibilities, all about the *shape* of the graph rather than its
contents:

1. Read the live schema (labels, properties, relationships) with a TTL
   cache. The cache invalidates after schema-evolution writes so the QA
   agent sees new labels immediately.
2. Create the base uniqueness constraints (idempotent).
3. Record an audit node every time the agent runs the schema-evolution
   workflow - this is the Command pattern's "applied command" log and is
   queryable from inside the graph.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import get_settings
from ..identifiers import safe_label, safe_property_name
from .session import Neo4jSession


BASE_CONSTRAINTS: list[tuple[str, str]] = [
    ("Customer", "id"),
    ("Location", "id"),
    ("Job", "id"),
    ("Source", "uri"),
    ("SchemaChangeProposal", "id"),
]


class SchemaRepository:
    def __init__(self, session: Neo4jSession) -> None:
        self.session = session
        self.ttl = get_settings().schema_cache_ttl_seconds
        self._cached: str | None = None
        self._cached_at: float = 0.0

    # ----- introspection ----------------------------------------------

    def get_schema(self, force_refresh: bool = False) -> str:
        now = time.monotonic()
        if (
            not force_refresh
            and self._cached is not None
            and (now - self._cached_at) < self.ttl
        ):
            return self._cached
        self.session.refresh_schema()
        self._cached = self.session.schema
        self._cached_at = now
        return self._cached

    def invalidate(self) -> None:
        self._cached = None
        self._cached_at = 0.0

    # ----- constraints ------------------------------------------------

    def ensure_base_constraints(self) -> None:
        for label, key in BASE_CONSTRAINTS:
            label = safe_label(label)
            key = safe_property_name(key)
            self.session.query(
                f"CREATE CONSTRAINT {label.lower()}_{key} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
            )

    def create_unique_constraint(self, label: str, key: str) -> None:
        label = safe_label(label)
        key = safe_property_name(key)
        self.session.query(
            f"CREATE CONSTRAINT {label.lower()}_{key} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        )

    # ----- audit / Command pattern ------------------------------------

    def record_proposal(
        self,
        *,
        proposal_id: str,
        activity_name: str,
        proposal: dict[str, Any],
        applied: bool,
    ) -> None:
        """Write a :SchemaChangeProposal audit node.

        Keeping the entire JSON on the node is fine for a demo; for a real
        deployment store it in the filestore and reference by uri.
        """
        self.session.query(
            """
            MERGE (p:SchemaChangeProposal {id: $id})
            SET p.activity_name = $activity_name,
                p.applied = $applied,
                p.applied_at = $applied_at,
                p.proposal_json = $proposal_json
            """,
            {
                "id": proposal_id,
                "activity_name": activity_name,
                "applied": applied,
                "applied_at": _utcnow(),
                "proposal_json": json.dumps(proposal, default=str),
            },
        )

    # ----- destructive ------------------------------------------------

    def wipe(self) -> None:
        self.session.query("MATCH (n) DETACH DELETE n")
        self.invalidate()


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
