"""Cypher-generating Q&A agent (modern replacement for ``GraphCypherQAChain``).

``langchain_neo4j.GraphCypherQAChain`` is deprecated for removal in
langchain-classic 1.0. The replacement pattern documented by both LangChain
and Neo4j is: hand the LLM a small set of Cypher tools (read schema, run
read-only query) and let it iterate inside an agent loop. That gives the
model the ability to look at a query error and try again instead of failing
the whole turn.

Two tools are exposed to the planner LLM:

* ``get_schema()`` - returns the live Neo4j schema (TTL-cached).
* ``run_cypher_readonly(cypher, params={})`` - executes the LLM's Cypher; if
  the query contains any write keyword the tool refuses.

The exact same skills file (``skills.py``) wraps this object's ``ask`` method
as ``answer_question`` so the outer chat agent can delegate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from .llm import build_llm
from .repositories import QueryRepository, SchemaRepository
from .repositories.query_repo import ReadOnlyViolation

CYPHER_SYSTEM_PROMPT = """\
You are an expert Neo4j 5 Cypher author working over a property graph.

How to operate:
1. Call `get_schema` ONCE at the start of each new question (results are
   cached upstream; one call per turn is enough).
2. Author Cypher that uses ONLY the labels, relationships, and properties
   in the schema. Never invent fields.
3. Prefer parameterised filters (e.g. `$status`) over string concatenation.
4. Handle nulls explicitly. For "last N days" use `date()` and
   `duration({{days: N}})`. For aggregations, alias columns clearly.
5. Do NOT return `prov_*` columns - those are internal.
6. Call `run_cypher_readonly` with your Cypher. If it errors, READ the error,
   fix the query, and try again. Up to 2 retries.
7. When you have rows, produce a concise natural-language answer that cites
   the rows (don't restate every row, summarize).
"""


@dataclass
class AgentAnswer:
    question: str
    cypher: str
    rows: list[dict[str, Any]]
    answer: str


class GraphQAAgent:
    """Agentic Q&A over the graph. One instance per process."""

    def __init__(
        self, schema_repo: SchemaRepository, query_repo: QueryRepository
    ) -> None:
        self.schema_repo = schema_repo
        self.query_repo = query_repo
        self._tools: list[BaseTool] = self._build_tools()
        self._agent = create_agent(
            model=build_llm(temperature=0.0),
            tools=self._tools,
            system_prompt=CYPHER_SYSTEM_PROMPT,
        )

    def _build_tools(self) -> list[BaseTool]:
        schema_repo = self.schema_repo
        query_repo = self.query_repo

        @tool
        def get_schema() -> str:
            """Return the live Neo4j schema (labels, relationships, properties)."""
            return schema_repo.get_schema()

        @tool
        def run_cypher_readonly(cypher: str, params: dict[str, Any] | None = None) -> str:
            """Execute a Cypher query against Neo4j and return rows as JSON.

            Refuses any query that contains write keywords (CREATE, MERGE,
            SET, DELETE, REMOVE, DETACH, DROP, FOREACH, or write-side APOC
            procedures). On any other error returns the error as a string so
            you can fix and retry.
            """
            try:
                rows = query_repo.run_readonly(cypher, params)
            except ReadOnlyViolation as e:
                return f"ReadOnlyViolation: {e}"
            except Exception as e:
                return f"CypherError: {type(e).__name__}: {e}"
            return json.dumps(rows, default=str, indent=2)

        return [get_schema, run_cypher_readonly]

    # ----- public API ---------------------------------------------------

    def ask(self, question: str) -> AgentAnswer:
        """Run the agent loop and return the final answer + last Cypher + rows."""
        result = self._agent.invoke(
            {
                "messages": [
                    SystemMessage(content=CYPHER_SYSTEM_PROMPT),
                    HumanMessage(content=question),
                ]
            }
        )
        messages = result.get("messages", [])

        cypher = ""
        rows: list[dict[str, Any]] = []
        final = ""
        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in msg.tool_calls or []:
                    if call.get("name") == "run_cypher_readonly":
                        cypher = call.get("args", {}).get("cypher", cypher) or cypher
                if msg.content:
                    final = _stringify(msg.content)
            elif isinstance(msg, ToolMessage) and msg.name == "run_cypher_readonly":
                payload = msg.content if isinstance(msg.content, str) else str(msg.content)
                try:
                    rows = json.loads(payload) if payload.startswith("[") else rows
                except Exception:
                    pass

        return AgentAnswer(question=question, cypher=cypher, rows=rows, answer=final.strip())


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c["text"] if isinstance(c, dict) and "text" in c else str(c) for c in content)
    return str(content)
