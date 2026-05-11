"""Provenance writes and lookups.

Every ingested entity links to a ``:Source`` node via ``WAS_DERIVED_FROM``;
this module owns the Cypher for both sides of that edge plus a uniform
"where did this node come from" lookup.
"""

from __future__ import annotations

import json
from typing import Any

from ..identifiers import safe_label, safe_property_name
from ..prov import SourceRecord
from .session import Neo4jSession


class ProvenanceRepository:
    def __init__(self, session: Neo4jSession) -> None:
        self.session = session

    def register_source(self, src: SourceRecord, *, derived_by: str) -> None:
        """Idempotent MERGE of a :Source node."""
        self.session.query(
            """
            MERGE (s:Source {uri: $uri})
            SET s.sha256 = $sha256,
                s.media_type = $media_type,
                s.ingested_at = $ingested_at,
                s.derived_by = $derived_by
            """,
            {
                "uri": src.uri,
                "sha256": src.sha256,
                "media_type": src.media_type,
                "ingested_at": src.ingested_at,
                "derived_by": derived_by,
            },
        )

    def list_sources(self) -> list[dict[str, Any]]:
        return self.session.query(
            """
            MATCH (s:Source)
            OPTIONAL MATCH (n)-[:WAS_DERIVED_FROM]->(s)
            WITH s, count(n) AS derived_count
            RETURN s.uri AS uri,
                   s.media_type AS media_type,
                   s.derived_by AS activity,
                   s.ingested_at AS ingested_at,
                   derived_count
            ORDER BY s.ingested_at DESC
            """
        )

    def lookup(self, label: str, key_property: str, key_value: str) -> list[dict[str, Any]]:
        """Return PROV-O properties for a node (validated label/property)."""
        label = safe_label(label)
        key_property = safe_property_name(key_property)
        return self.session.query(
            f"""
            MATCH (n:{label} {{{key_property}: $value}})-[:WAS_DERIVED_FROM]->(s:Source)
            RETURN n.cid AS cid,
                   n.prov_derived_by AS derived_by,
                   n.prov_ingested_at AS ingested_at,
                   s.uri AS source_uri,
                   s.sha256 AS source_sha256,
                   s.media_type AS media_type
            """,
            {"value": key_value},
        )

    def dump_as_json(self, rows: list[dict[str, Any]]) -> str:
        return json.dumps(rows, default=str, indent=2)
