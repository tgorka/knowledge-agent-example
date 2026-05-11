"""Data import workflow.

Two entry points:

* :meth:`Pipeline.import_seed` - deterministic ingest of well-formed seed
  JSON (customers / locations / jobs). No LLM involved.
* :meth:`Pipeline.evolve_and_ingest` - the schema-evolution path. Given a
  folder of *messy* payloads, the agent proposes a schema extension (new
  labels, relationships, properties), an injected ``approver`` callable
  approves or declines, and on approval we apply the schema and ingest the
  data.

Design notes:

* **Anti-corruption layer.** Every identifier (label, relationship type,
  property key) the LLM produces is run through ``identifiers.safe_*``
  before being interpolated into Cypher. The identifier rules are strict
  ASCII so the failure mode is loud rather than dangerous.
* **Command pattern.** Each evolution pass writes a
  ``:SchemaChangeProposal`` audit node with the full proposal JSON and an
  ``applied`` flag. This is the queryable record of "what did the agent
  ever propose, and what did we let it through".
* **UNWIND-batched writes.** Extracted nodes and edges are grouped by label
  / relationship type and written in one round-trip per group, rather than
  per-entity.
* **No rich.Console.** This module logs via stdlib ``logging``; rendering
  happens in the CLI / chat layer.

Every node (seeded or evolved) carries PROV-O style provenance properties
and links back to a ``:Source`` node so we always know which raw file
produced it.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .config import get_settings
from .identifiers import safe_label, safe_property_name, safe_relationship_type
from .llm import build_llm
from .prov import SourceRecord, make_content_id, prov_properties, store_payload
from .repositories import (
    Neo4jSession,
    ProvenanceRepository,
    QueryRepository,
    SchemaRepository,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for the LLM's structured output
# ---------------------------------------------------------------------------


class ProposedLabel(BaseModel):
    label: str
    properties: list[str]
    unique_key: str | None = None
    rationale: str


class ProposedRelationship(BaseModel):
    type: str
    from_label: str
    to_label: str
    properties: list[str] = Field(default_factory=list)
    rationale: str


class SchemaProposal(BaseModel):
    new_labels: list[ProposedLabel]
    new_relationships: list[ProposedRelationship]
    normalization_rules: list[str]
    summary: str


class ExtractedNode(BaseModel):
    label: str
    key_property: str
    key_value: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedEdge(BaseModel):
    type: str
    from_label: str
    from_key: str
    from_value: str
    to_label: str
    to_key: str
    to_value: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedBatch(BaseModel):
    nodes: list[ExtractedNode]
    edges: list[ExtractedEdge]
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class RawPayload:
    filename: str
    text: str


@dataclass
class EvolutionResult:
    """Returned from ``evolve_and_ingest``. Always carries the proposal; the
    ``applied`` flag tells the caller whether the approver let it through."""

    proposal: SchemaProposal
    proposal_id: str
    applied: bool
    activity_name: str


Approver = Callable[[SchemaProposal], bool]


def always_approve(_: SchemaProposal) -> bool:
    return True


class Pipeline:
    """Orchestrates seed import and the schema-evolution workflow."""

    def __init__(
        self,
        session: Neo4jSession | None = None,
        *,
        schema_repo: SchemaRepository | None = None,
        query_repo: QueryRepository | None = None,
        prov_repo: ProvenanceRepository | None = None,
    ) -> None:
        self.session = session or Neo4jSession()
        self.schema_repo = schema_repo or SchemaRepository(self.session)
        self.query_repo = query_repo or QueryRepository(self.session)
        self.prov_repo = prov_repo or ProvenanceRepository(self.session)
        self.settings = get_settings()
        # Two LLM handles: a "chat-size" one for the default 1024-token cap,
        # and a "structured-output" one with a larger cap for proposal +
        # extraction JSON. Splitting per-task keeps chat cheap while the
        # evolution path still has room to emit a full ExtractedBatch.
        self.llm = build_llm(temperature=0.0)
        self.llm_structured = build_llm(
            temperature=0.0,
            max_tokens=self.settings.llm_max_tokens_structured,
        )

    # ----- seed --------------------------------------------------------

    def import_seed(self, seed_dir: Path) -> None:
        """Deterministic ingest of customers, locations, and jobs."""
        self.schema_repo.ensure_base_constraints()

        for filename, label, key in [
            ("customers.json", "Customer", "id"),
            ("locations.json", "Location", "id"),
            ("jobs.json", "Job", "id"),
        ]:
            src = store_payload(seed_dir / filename)
            self.prov_repo.register_source(src, derived_by="seed_import")
            rows = json.loads((seed_dir / filename).read_text())
            self._write_seed_nodes(label, key, rows, src)

        self._link_seed_relationships()
        self.schema_repo.invalidate()
        log.info("seed import complete")

    def _write_seed_nodes(
        self, label: str, key: str, rows: list[dict[str, Any]], src: SourceRecord
    ) -> None:
        label = safe_label(label)
        key = safe_property_name(key)
        prov = prov_properties(src, derived_by="seed_import")
        enriched = []
        for row in rows:
            row = dict(row)
            row["cid"] = make_content_id(label, row[key])
            row.update(prov)
            enriched.append(row)
        self.query_repo.write_batch(
            f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{{key}: row.{key}}})
            SET n += row
            WITH n, row
            MATCH (s:Source {{uri: row.prov_source_uri}})
            MERGE (n)-[:WAS_DERIVED_FROM]->(s)
            """,
            enriched,
        )

    def _link_seed_relationships(self) -> None:
        self.query_repo.run(
            """
            MATCH (c:Customer), (l:Location {owner: c.id})
            MERGE (c)-[:OWNS]->(l)
            """
        )
        self.query_repo.run(
            """
            MATCH (j:Job), (c:Customer {id: j.customer}), (l:Location {id: j.location})
            MERGE (c)-[:REQUESTED]->(j)
            MERGE (j)-[:AT_LOCATION]->(l)
            """
        )

    # ----- schema evolution --------------------------------------------

    def evolve_and_ingest(
        self,
        payloads_dir: Path,
        activity_name: str,
        approver: Approver | None = None,
    ) -> EvolutionResult:
        """Propose a schema, ask the approver, then apply + ingest on a yes.

        The ``approver`` callable lets the caller plug in whatever HITL flow
        they like - ``rich.Confirm.ask`` for the CLI, a LangGraph
        ``interrupt()``-backed function for the chat agent, ``always_approve``
        for non-interactive tests.
        """
        approver = approver or self._default_approver()
        payloads = self._load_payloads(payloads_dir)
        proposal = self._propose_schema(payloads)
        proposal_id = str(uuid.uuid4())

        approved = approver(proposal)
        # Always record the proposal, even if declined - that's the audit log.
        self.schema_repo.record_proposal(
            proposal_id=proposal_id,
            activity_name=activity_name,
            proposal=proposal.model_dump(),
            applied=approved,
        )
        if not approved:
            log.info("evolution declined: %s", activity_name)
            return EvolutionResult(
                proposal=proposal,
                proposal_id=proposal_id,
                applied=False,
                activity_name=activity_name,
            )

        self._apply_schema(proposal)

        for payload in payloads:
            src = store_payload(payloads_dir / payload.filename)
            self.prov_repo.register_source(src, derived_by=activity_name)
            extracted = self._extract_entities(payload, proposal)
            self._ingest_extracted(extracted, src, activity_name)

        self.schema_repo.invalidate()
        log.info("evolution applied: %s", activity_name)
        return EvolutionResult(
            proposal=proposal,
            proposal_id=proposal_id,
            applied=True,
            activity_name=activity_name,
        )

    def _default_approver(self) -> Approver:
        """If env says approval is required, refuse silently when no approver
        is wired in. That's safer than auto-applying."""
        if self.settings.schema_evolution_require_approval:
            def _refuse(_: SchemaProposal) -> bool:
                log.warning(
                    "schema evolution refused: no approver wired in and "
                    "SCHEMA_EVOLUTION_REQUIRE_APPROVAL=true"
                )
                return False
            return _refuse
        return always_approve

    def _load_payloads(self, payloads_dir: Path) -> list[RawPayload]:
        files = sorted(p for p in payloads_dir.iterdir() if p.is_file())
        return [
            RawPayload(filename=p.name, text=p.read_text(encoding="utf-8"))
            for p in files
        ]

    # ----- LLM steps ---------------------------------------------------

    def _propose_schema(self, payloads: list[RawPayload]) -> SchemaProposal:
        current_schema = self.schema_repo.get_schema()
        joined = "\n\n".join(f"# {p.filename}\n{p.text}" for p in payloads)
        prompt = f"""
You are a senior knowledge-graph engineer. The Neo4j graph below already exists.

CURRENT SCHEMA
{current_schema}

NEW RAW PAYLOADS (mixed formats - JSON, email, CSV, free text):
{joined}

Your task:
1. Decide which new node labels and relationships are needed to represent these
   payloads in the graph. Prefer separate nodes (not properties) when an entity
   has its own lifecycle, identifiers, or 1:N relationships. For each, give a
   short rationale.
2. Suggest which existing labels they should connect to (Customer, Location,
   Job, Source).
3. List the normalization rules a deterministic ingester will need to apply
   (e.g. "parse dates of form M/D/YY as %m/%d/%y", "fees with $ stripped").

Identifier rules (strict): labels, relationship types, and property names you
propose MUST match [A-Za-z_][A-Za-z0-9_]* (ASCII, no spaces, no dashes). Any
identifier that fails this will be rejected by the ingester.
""".strip()
        # ``method="function_calling"`` instead of the default strict
        # ``json_schema`` mode: our Pydantic models include free-form
        # ``dict[str, Any]`` property bags which strict mode rejects
        # ("additionalProperties must be false"). Function-calling has more
        # lenient schema rules and is what the LangChain migration guide
        # recommends for schemas that aren't strict-mode-clean.
        structured = self.llm_structured.with_structured_output(
            SchemaProposal, method="function_calling"
        )
        return structured.invoke(prompt)

    def _apply_schema(self, proposal: SchemaProposal) -> None:
        for lbl in proposal.new_labels:
            label = safe_label(lbl.label)
            if lbl.unique_key:
                self.schema_repo.create_unique_constraint(label, lbl.unique_key)

    def _extract_entities(self, payload: RawPayload, proposal: SchemaProposal) -> ExtractedBatch:
        labels_text = json.dumps([lbl.model_dump() for lbl in proposal.new_labels], indent=2)
        rels_text = json.dumps([r.model_dump() for r in proposal.new_relationships], indent=2)
        prompt = f"""
You are extracting structured graph data from one raw payload.

NEW LABELS YOU MAY CREATE:
{labels_text}

NEW RELATIONSHIPS YOU MAY CREATE:
{rels_text}

YOU MAY ALSO REFERENCE EXISTING LABELS: Customer, Location, Job (match by their `id` property).

NORMALIZATION RULES (apply them):
{json.dumps(proposal.normalization_rules)}

RAW PAYLOAD ({payload.filename}):
{payload.text}

Return STRICT JSON describing every node and edge. Use ISO dates (YYYY-MM-DD).
If a value is unknown, omit the property rather than inventing one.
Labels, relationship types, and property names MUST match [A-Za-z_][A-Za-z0-9_]*.
""".strip()
        # See note in ``_propose_schema``: function-calling mode is required
        # because ``ExtractedNode.properties`` / ``ExtractedEdge.properties``
        # are open dicts, which strict ``json_schema`` mode refuses.
        structured = self.llm_structured.with_structured_output(
            ExtractedBatch, method="function_calling"
        )
        return structured.invoke(prompt)

    # ----- UNWIND-batched ingest --------------------------------------

    def _ingest_extracted(
        self, batch: ExtractedBatch, src: SourceRecord, activity: str
    ) -> None:
        prov = prov_properties(src, derived_by=activity)
        self._ingest_nodes(batch.nodes, src, prov)
        self._ingest_edges(batch.edges)

    def _ingest_nodes(
        self,
        nodes: list[ExtractedNode],
        src: SourceRecord,
        prov: dict[str, Any],
    ) -> None:
        # Group by (label, key_property) so each group gets a single UNWIND.
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for n in nodes:
            label = safe_label(n.label)
            key_prop = safe_property_name(n.key_property)
            row = {
                "key": n.key_value,
                "props": {
                    **n.properties,
                    key_prop: n.key_value,
                    "cid": make_content_id(label, str(n.key_value)),
                    **prov,
                },
            }
            groups.setdefault((label, key_prop), []).append(row)

        for (label, key_prop), rows in groups.items():
            # prov_source_uri is already on every row.props - the MATCH below
            # uses that rather than a separate $source_uri param so we can
            # keep using the ``$rows``-only write_batch helper.
            self.query_repo.write_batch(
                f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{{key_prop}: row.key}})
                SET n += row.props
                WITH n, row
                MATCH (s:Source {{uri: row.props.prov_source_uri}})
                MERGE (n)-[:WAS_DERIVED_FROM]->(s)
                """,
                rows,
            )

    def _ingest_edges(self, edges: list[ExtractedEdge]) -> None:
        # Group by (from_label, from_key, rel_type, to_label, to_key).
        groups: dict[
            tuple[str, str, str, str, str], list[dict[str, Any]]
        ] = {}
        for e in edges:
            key = (
                safe_label(e.from_label),
                safe_property_name(e.from_key),
                safe_relationship_type(e.type),
                safe_label(e.to_label),
                safe_property_name(e.to_key),
            )
            groups.setdefault(key, []).append(
                {
                    "from_value": e.from_value,
                    "to_value": e.to_value,
                    "props": e.properties or {},
                }
            )

        for (from_label, from_key, rel_type, to_label, to_key), rows in groups.items():
            self.query_repo.write_batch(
                f"""
                UNWIND $rows AS row
                MATCH (a:{from_label} {{{from_key}: row.from_value}})
                MATCH (b:{to_label} {{{to_key}: row.to_value}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += row.props
                """,
                rows,
            )


