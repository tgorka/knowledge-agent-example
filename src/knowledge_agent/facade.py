"""Composition root.

One object that owns the Neo4j session, the three repositories, the import
pipeline, and the QA agent. The CLI / chat / skills layers all talk to this
- the rest of the package stays unaware of who's driving.
"""

from __future__ import annotations

from pathlib import Path

from .pipeline import EvolutionResult, Pipeline, SchemaProposal
from .pipeline import Approver
from .qa import AgentAnswer, GraphQAAgent
from .repositories import (
    Neo4jSession,
    ProvenanceRepository,
    QueryRepository,
    SchemaRepository,
)


class KnowledgeAgent:
    def __init__(self) -> None:
        self.session = Neo4jSession()
        self.schema_repo = SchemaRepository(self.session)
        self.query_repo = QueryRepository(self.session)
        self.prov_repo = ProvenanceRepository(self.session)
        self._pipeline: Pipeline | None = None
        self._qa: GraphQAAgent | None = None

    # ----- lazy components --------------------------------------------

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = Pipeline(
                self.session,
                schema_repo=self.schema_repo,
                query_repo=self.query_repo,
                prov_repo=self.prov_repo,
            )
        return self._pipeline

    @property
    def qa(self) -> GraphQAAgent:
        if self._qa is None:
            self._qa = GraphQAAgent(self.schema_repo, self.query_repo)
        return self._qa

    # ----- verbs -------------------------------------------------------

    def reset(self) -> None:
        self.schema_repo.wipe()

    def seed(self, seed_dir: Path) -> None:
        self.pipeline.import_seed(seed_dir)

    def evolve(
        self,
        payloads_dir: Path,
        activity_name: str = "schema_evolution",
        approver: Approver | None = None,
    ) -> EvolutionResult:
        return self.pipeline.evolve_and_ingest(
            payloads_dir, activity_name=activity_name, approver=approver
        )

    def ask(self, question: str) -> AgentAnswer:
        return self.qa.ask(question)

    def close(self) -> None:
        self.session.close()


__all__ = ["KnowledgeAgent", "SchemaProposal", "EvolutionResult", "Approver", "AgentAnswer"]
