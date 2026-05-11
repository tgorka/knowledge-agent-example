"""Run arbitrary parameterised Cypher.

Only used in two places:

* The QA agent's ``run_cypher_readonly`` tool, which feeds queries that the
  LLM authored.
* Internal helpers that build their own dynamic-but-validated Cypher (e.g.
  the pipeline's UNWIND-batched ingest helpers).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .session import Neo4jSession


# Conservative write-keyword guard for the read-only tool. We strip string
# literals before scanning so a query that simply *contains* the word "MERGE"
# inside a parameter value is still allowed. False negatives are still
# possible (e.g. APOC procedures that write) but it's good enough for a demo
# where the worst case is the LLM accidentally running a UNWIND-MERGE.
_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH|DROP|FOREACH|CALL\s+apoc\.(create|merge|refactor)\b)",
    re.IGNORECASE,
)
_STRING_LIT = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


class ReadOnlyViolation(ValueError):
    """Raised when ``run_readonly`` is asked to execute a write query."""


class QueryRepository:
    def __init__(self, session: Neo4jSession) -> None:
        self.session = session

    def run(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.session.query(cypher, params or {})

    def run_readonly(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Reject anything that smells like a write before executing."""
        stripped = _STRING_LIT.sub("''", cypher)
        if _WRITE_KEYWORDS.search(stripped):
            raise ReadOnlyViolation(
                f"Refusing to run a query that contains write keywords: {cypher!r}"
            )
        return self.session.query(cypher, params or {})

    def write_batch(self, cypher: str, rows: Iterable[dict[str, Any]]) -> None:
        self.session.write_batch(cypher, rows)
