"""Anti-corruption layer for Cypher identifiers - pure-function tests.

These run with no Neo4j / OpenRouter credentials. The regex is the only
thing standing between an LLM-supplied label like ``Customer) DETACH DELETE n //``
and our Cypher executor, so the asserts here are deliberately paranoid.
"""

from __future__ import annotations

import pytest

from knowledge_agent.identifiers import (
    CypherIdentifierError,
    safe_label,
    safe_property_name,
    safe_relationship_type,
)


VALID = ["Customer", "_internal", "Permit_v2", "A1", "x" * 64]
INVALID = [
    "",
    "1Customer",  # starts with digit
    "Customer-Type",  # dash
    "Customer Type",  # space
    "Customer;",  # semicolon
    "Customer) DETACH DELETE n //",  # the canonical injection payload
    "Customer\nDELETE",  # newline
    "x" * 65,  # too long
    "café",  # non-ASCII
    "`Customer`",  # backticks
    None,
    123,
]


@pytest.mark.parametrize("value", VALID)
def test_safe_label_accepts_clean_identifiers(value: str) -> None:
    assert safe_label(value) == value
    assert safe_relationship_type(value) == value
    assert safe_property_name(value) == value


@pytest.mark.parametrize("value", INVALID)
def test_safe_label_rejects_bad_input(value) -> None:
    with pytest.raises(CypherIdentifierError):
        safe_label(value)
    with pytest.raises(CypherIdentifierError):
        safe_relationship_type(value)
    with pytest.raises(CypherIdentifierError):
        safe_property_name(value)
