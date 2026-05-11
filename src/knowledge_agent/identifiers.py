"""Anti-corruption layer for Cypher identifiers.

Neo4j parameters (``$foo``) can bind property *values* but not labels,
relationship types, or property *keys*. Anything that interpolates one of
those into a Cypher string at runtime is a potential injection vector - and
in this project that includes every f-string in ``pipeline.py`` and
``skills.show_provenance`` because labels/relationship types come from the
LLM's structured output or, worse, from agent tool arguments.

Two defenses are in use:

1. **Validation** - this module. Every identifier coming from the LLM or
   from a tool argument must pass ``safe_label`` / ``safe_relationship_type``
   / ``safe_property_name`` before being inserted into Cypher. Anything that
   doesn't match a strict regex raises ``CypherIdentifierError``.
2. **Dynamic identifier syntax** - Neo4j 5.14+ supports ``:$(label)`` and
   ``[:$(rel)]`` which take the value as a normal parameter and let the
   server validate it. Where the surrounding query allows it we prefer this;
   where it doesn't (e.g. ``CREATE CONSTRAINT``), validation is the only
   line of defense.

The regex is intentionally narrow: ASCII letters, digits, underscore, with a
leading letter or underscore, max 64 characters. Neo4j accepts more (Unicode
identifiers, backticked names) but for a demo where the LLM is the producer
the narrow rule is fine and the failures are loud.
"""

from __future__ import annotations

import re

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class CypherIdentifierError(ValueError):
    """Raised when an identifier headed for Cypher fails validation."""


def _check(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _IDENT.fullmatch(value):
        raise CypherIdentifierError(
            f"invalid {kind} {value!r}: must match {_IDENT.pattern}"
        )
    return value


def safe_label(value: str) -> str:
    """Validate a Neo4j node label. Returns the value unchanged on success."""
    return _check("label", value)


def safe_relationship_type(value: str) -> str:
    """Validate a Neo4j relationship type. Returns the value unchanged on success."""
    return _check("relationship type", value)


def safe_property_name(value: str) -> str:
    """Validate a property key. Returns the value unchanged on success."""
    return _check("property name", value)
