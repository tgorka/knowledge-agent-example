"""LangChain skills (a.k.a. tools) exposing each use case in the brief.

The brief calls for an agent that can (a) read the schema and answer NL
questions, (b) evolve the schema when given messy new data, and (c) re-query
across the original and newly added entities. Rather than encode that flow
once and forever in Python, we surface it as a small set of well-described
``@tool`` functions and let the agent compose them.

Use cases -> skills:

* "Read the schema and write Cypher to answer NL questions" (Part 1)
    -> ``inspect_schema``, ``answer_question``
* "Look at messy payloads, propose new labels/relationships, justify them"
    -> ``propose_schema_evolution``
* "Apply the schema and ingest the data" (Part 2 #2 + #3)
    -> ``apply_schema_evolution`` (uses LangGraph ``interrupt()`` for HITL)
* "Re-query across original + new entities" (Part 2 #4)
    -> ``answer_question`` (same tool, refreshed schema)
* "Trace a fact back to its raw payload" (PROV-O hook)
    -> ``show_provenance``, ``list_sources``
* "Audit the schema-evolution history" (Command pattern audit log)
    -> ``list_schema_changes``
* "Bootstrap / reset"
    -> ``seed_base_data``, ``reset_graph``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from .facade import KnowledgeAgent
from .pipeline import SchemaProposal


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def build_skills(ka: KnowledgeAgent) -> list[BaseTool]:
    """Return the skill set bound to a live ``KnowledgeAgent`` instance.

    The ``ka`` reference is captured in closures - we deliberately don't make
    these tools accept a connection string, because the agent shouldn't be
    reasoning about credentials.
    """

    @tool
    def inspect_schema() -> str:
        """Return the live Neo4j schema (labels, properties, relationships).

        Always call this first if you are about to generate Cypher or reason
        about what data exists. The schema cache invalidates automatically
        after ``apply_schema_evolution`` runs.
        """
        return ka.schema_repo.get_schema()

    @tool
    def answer_question(question: str) -> str:
        """Answer a natural-language question against the graph.

        Generates Cypher from the current schema, executes it, and returns
        the rows plus a natural-language summary. Use this for any analytical
        or lookup question - "which customers...", "average ticket size...",
        "permits expiring soon...", etc.
        """
        a = ka.ask(question)
        return json.dumps(
            {
                "question": a.question,
                "cypher": a.cypher,
                "row_count": len(a.rows),
                "rows_preview": a.rows[:10],
                "answer": a.answer,
            },
            default=str,
            indent=2,
        )

    @tool
    def list_sources() -> str:
        """List every raw payload (``:Source``) that has been ingested."""
        return ka.prov_repo.dump_as_json(ka.prov_repo.list_sources())

    @tool
    def show_provenance(label: str, key_property: str, key_value: str) -> str:
        """Return PROV-O metadata for a specific node.

        Example: ``show_provenance("Customer", "id", "c1")`` or
        ``show_provenance("Permit", "permit_no", "AUS-MECH-2026-04881")``.
        Returns source URI, sha256, ingest time, and which pipeline activity
        wrote the node. ``label`` and ``key_property`` must be valid Neo4j
        identifiers; anything weird will be refused.
        """
        rows = ka.prov_repo.lookup(label, key_property, key_value)
        if not rows:
            return f"No node of label {label} with {key_property}={key_value!r}."
        return ka.prov_repo.dump_as_json(rows)

    @tool
    def propose_schema_evolution(payloads_dir: str = "data/permits") -> str:
        """Read raw payloads and propose a schema extension. READ-ONLY.

        Returns the proposal as JSON (new labels with rationale, new
        relationships, normalization rules). Use this when the user hands you
        data that doesn't obviously fit the current schema. To then apply
        the proposal, call ``apply_schema_evolution``.
        """
        from .pipeline import RawPayload

        target = _resolve_dir(payloads_dir)
        payloads = [
            RawPayload(filename=p.name, text=p.read_text(encoding="utf-8"))
            for p in sorted(target.iterdir())
            if p.is_file()
        ]
        proposal = ka.pipeline._propose_schema(payloads)
        return proposal.model_dump_json(indent=2)

    @tool
    def apply_schema_evolution(
        payloads_dir: str = "data/permits",
        activity_name: str = "schema_evolution_v1",
    ) -> str:
        """Propose + apply + ingest from a folder of raw payloads.

        HUMAN APPROVAL REQUIRED. This tool uses LangGraph's ``interrupt()``
        to pause the agent and surface the proposal to the operator; the
        operator resumes with ``"approve"`` or ``"decline"``. On decline,
        nothing is written to the graph but the proposal is still recorded
        as a ``:SchemaChangeProposal`` audit node with ``applied=false``.
        """
        target = _resolve_dir(payloads_dir)

        def _interrupt_approver(proposal: SchemaProposal) -> bool:
            decision = interrupt(
                {
                    "kind": "schema_evolution_approval",
                    "activity_name": activity_name,
                    "payloads_dir": str(target),
                    "proposal": proposal.model_dump(),
                }
            )
            return _interpret_decision(decision)

        result = ka.evolve(
            target, activity_name=activity_name, approver=_interrupt_approver
        )
        return json.dumps(
            {
                "applied": result.applied,
                "activity_name": result.activity_name,
                "proposal_id": result.proposal_id,
                "summary": result.proposal.summary,
            },
            indent=2,
        )

    @tool
    def list_schema_changes() -> str:
        """List the audit log of schema-change proposals.

        Returns every ``:SchemaChangeProposal`` node the pipeline has
        written, oldest first, with whether it was applied. This is how the
        operator answers "what has the agent ever tried to change about the
        graph schema".
        """
        rows = ka.query_repo.run(
            """
            MATCH (p:SchemaChangeProposal)
            RETURN p.id AS id,
                   p.activity_name AS activity_name,
                   p.applied AS applied,
                   p.applied_at AS applied_at
            ORDER BY p.applied_at ASC
            """
        )
        return json.dumps(rows, default=str, indent=2)

    @tool
    def seed_base_data() -> str:
        """Load the deterministic seed data (customers, locations, jobs).

        Idempotent - calling twice is safe; MERGE keys prevent duplicates.
        Use this once after ``reset_graph`` to bootstrap Part 1.
        """
        ka.seed(DATA_ROOT / "seed")
        return "Seed import complete: Customer, Location, Job nodes loaded."

    @tool
    def reset_graph(confirm: str) -> str:
        """Wipe ALL nodes and relationships from the graph. Destructive.

        Requires ``confirm="yes-wipe"`` as a guard against accidental calls.
        """
        if confirm != "yes-wipe":
            return "Refused: pass confirm='yes-wipe' to actually wipe the graph."
        ka.reset()
        return "Graph wiped."

    return [
        inspect_schema,
        answer_question,
        list_sources,
        show_provenance,
        propose_schema_evolution,
        apply_schema_evolution,
        list_schema_changes,
        seed_base_data,
        reset_graph,
    ]


def _interpret_decision(decision: Any) -> bool:
    """Translate whatever the operator resumed with into a boolean."""
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, str):
        return decision.strip().lower() in {"approve", "yes", "y", "true", "ok"}
    if isinstance(decision, dict):
        return _interpret_decision(decision.get("approve", decision.get("decision", False)))
    return False


def _resolve_dir(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = DATA_ROOT.parent / path
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return path


def describe_skills(skills: list[BaseTool] | None = None) -> str:
    """Pretty-print skill names + descriptions, used by `knowledge-agent skills`.

    Passing ``None`` builds the skill list against a stub agent so we can list
    them without actually connecting to Neo4j.
    """
    skills = skills if skills is not None else build_skills(_StubAgent())  # type: ignore[arg-type]
    lines: list[str] = []
    for s in skills:
        desc = (s.description or "").strip().splitlines()[0] if s.description else ""
        lines.append(f"  - {s.name}: {desc}")
    return "\n".join(lines)


class _StubAgent:
    """No-connection placeholder for ``describe_skills(None)``."""

    schema_repo = None
    query_repo = None
    prov_repo = None
    pipeline = None

    def seed(self, *_: object, **__: object) -> None: ...
    def reset(self, *_: object, **__: object) -> None: ...
    def ask(self, *_: object, **__: object) -> None: ...
    def evolve(self, *_: object, **__: object) -> None: ...
