"""Repository pattern over Neo4j.

The repositories own all Cypher in the codebase. They share one
``langchain_neo4j.Neo4jGraph`` connection wrapped by :class:`Neo4jSession`
(``session.py``) and expose narrow, intention-revealing methods to the rest
of the package - no f-strung Cypher leaks out of here.

* :class:`SchemaRepository` - reads the live schema with a TTL cache, creates
  constraints, and records ``:SchemaChangeProposal`` audit nodes.
* :class:`QueryRepository` - parameterised reads and writes for arbitrary
  Cypher; the only place ``Neo4jGraph.query`` is called directly.
* :class:`ProvenanceRepository` - writes :Source nodes, links derived
  entities via ``WAS_DERIVED_FROM``, and answers "where did this come from".
"""

from .session import Neo4jSession
from .schema_repo import SchemaRepository
from .query_repo import QueryRepository
from .prov_repo import ProvenanceRepository

__all__ = [
    "Neo4jSession",
    "SchemaRepository",
    "QueryRepository",
    "ProvenanceRepository",
]
