"""Provenance helpers.

We use a thin PROV-O slice (Entity / Activity / Agent / wasDerivedFrom) plus
deterministic content-addressed identifiers (``cid``) so every node is
identifiable independent of its mutable label/property bag. The original
payload is written to a local ``filestore`` that stands in for an S3 bucket
(see README).

Note: earlier revisions of this module called the identifier a "Wikidata-style
Q-number" - that was inaccurate. Wikidata Q-numbers are monotonically issued
counters; ours is a sha1-derived content hash with a ``Q`` prefix kept only
for visual brevity.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_settings


@dataclass(frozen=True)
class SourceRecord:
    """A pointer to the raw payload backing one or more graph entities."""

    uri: str
    sha256: str
    media_type: str
    ingested_at: str  # ISO-8601 UTC


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_content_id(namespace: str, natural_key: str) -> str:
    """Deterministic content-addressed id, stable across re-imports.

    Format: ``Q`` + first 12 hex chars of ``sha1(namespace:natural_key)``.
    The leading ``Q`` is purely cosmetic so the id reads like a Wikidata-style
    handle; it is NOT a Wikidata identifier. 12 hex chars is collision-
    resistant for a demo dataset and short enough to eyeball in the console.
    """
    h = hashlib.sha1(f"{namespace}:{natural_key}".encode("utf-8")).hexdigest()
    return "Q" + h[:12]


def store_payload(source_path: Path, payload_bytes: bytes | None = None) -> SourceRecord:
    """Copy (or write) a raw payload into the filestore and return its prov record.

    ``source_path`` is the *original* path on disk; the file is copied verbatim
    into the filestore under the same name. If ``payload_bytes`` is supplied we
    use it instead of re-reading the source - useful for synthesized payloads.
    """
    settings = get_settings()
    target = settings.filestore_dir / source_path.name
    target.parent.mkdir(parents=True, exist_ok=True)

    if payload_bytes is None:
        shutil.copyfile(source_path, target)
        payload_bytes = target.read_bytes()
    else:
        target.write_bytes(payload_bytes)

    sha = hashlib.sha256(payload_bytes).hexdigest()
    media_type = _guess_media_type(source_path.suffix)
    uri = f"{settings.filestore_bucket}://{source_path.name}"

    return SourceRecord(uri=uri, sha256=sha, media_type=media_type, ingested_at=_utcnow_iso())


def _guess_media_type(suffix: str) -> str:
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".txt": "text/plain",
    }.get(suffix.lower(), "application/octet-stream")


def prov_properties(source: SourceRecord, derived_by: str) -> dict[str, Any]:
    """Properties to attach to every entity ingested from a given source.

    ``derived_by`` names the activity (e.g. ``"seed_import"`` or
    ``"permit_evolution_v1"``) so we can audit which pipeline run touched what.
    """
    return {
        "prov_source_uri": source.uri,
        "prov_source_sha256": source.sha256,
        "prov_media_type": source.media_type,
        "prov_ingested_at": source.ingested_at,
        "prov_derived_by": derived_by,
    }


def write_text_payload(filename: str, text: str) -> SourceRecord:
    """Convenience for storing a payload we already have in memory (e.g. an email)."""
    settings = get_settings()
    target = settings.filestore_dir / filename
    target.write_text(text, encoding="utf-8")
    return store_payload(target, text.encode("utf-8"))


def write_json_payload(filename: str, obj: Any) -> SourceRecord:
    return write_text_payload(filename, json.dumps(obj, indent=2, sort_keys=True))
