"""Interactive chat REPL backed by the ``skills`` toolset.

The user types in natural language, ``langchain.agents.create_agent`` decides
which skill(s) to call, and we render the conversation. This is the
"agent uses tools" demo path - the deterministic CLI subcommands (``seed``,
``evolve``, ``ask``) still exist for showing the underlying mechanics.

HITL approval flow:

* When the agent calls ``apply_schema_evolution``, the skill body calls
  LangGraph's ``interrupt()`` with the proposal as payload.
* ``agent.invoke`` then returns a dict that includes ``__interrupt__``.
* We render the proposal in a rich panel, ask the operator y/n, and resume
  via ``agent.invoke(Command(resume=decision), config=...)``.

This requires a checkpointer (``InMemorySaver`` is fine for a single REPL
session) and a stable ``thread_id`` to identify the paused run.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax

from .facade import KnowledgeAgent
from .llm import build_llm
from .skills import build_skills, describe_skills

console = Console()


SYSTEM_PROMPT = """\
You are a knowledge-graph assistant over a Neo4j FSM (field-service-management)
database. You have a small set of skills available and MUST use them rather
than answering from memory.

How to operate:
1. Always call `inspect_schema` first when you are about to write Cypher or
   reason about what data exists. The schema can change between turns when
   `apply_schema_evolution` runs.
2. For any analytical or lookup question, call `answer_question` and pass the
   user's question verbatim - it generates and executes the Cypher for you.
3. If the user gives you a folder of new payloads or asks to "ingest" /
   "extend the schema", first call `propose_schema_evolution` to show the
   plan. Only call `apply_schema_evolution` if the user has agreed (or asked
   you to "do it"/"go ahead"). The apply tool will surface the proposal to
   the operator for approval before any writes happen.
4. When the user asks "where did X come from", use `show_provenance` or
   `list_sources` rather than guessing.
5. When the user asks "what schema changes happened" or "audit", use
   `list_schema_changes`.
6. Be terse. Cite the rows you saw, not just the prose answer.
"""

# Tool names whose outputs are short and useful to render in full.
_FULL_OUTPUT_TOOLS = {
    "inspect_schema",
    "show_provenance",
    "list_sources",
    "list_schema_changes",
    "propose_schema_evolution",
    "reset_graph",
    "seed_base_data",
}
# Outputs from these tools can be large (extracted batches, NL answers); we
# truncate them for readability.
_TRUNCATE_AT = 2000


def run_chat(ka: KnowledgeAgent | None = None) -> None:
    own = ka is None
    ka = ka or KnowledgeAgent()
    try:
        skills = build_skills(ka)
        checkpointer = InMemorySaver()
        agent = create_agent(
            model=build_llm(),
            tools=skills,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        console.print(
            Panel(
                "Skills available:\n" + describe_skills(skills) + "\n\n"
                "Type a question or instruction. `exit` / Ctrl-D to quit.",
                title="knowledge-agent chat",
                border_style="cyan",
            )
        )

        while True:
            try:
                user_in = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye.[/dim]")
                return
            if not user_in:
                continue
            if user_in.lower() in {"exit", "quit", ":q"}:
                return

            _drive_turn(agent, config, input_=HumanMessage(content=user_in))
    finally:
        if own:
            ka.close()


def _drive_turn(agent: Any, config: dict, *, input_: Any) -> None:
    """Run one user-message-to-final-answer cycle, handling interrupts.

    LangGraph's tool-level ``interrupt()`` causes ``invoke`` to return early
    with an ``__interrupt__`` key. We surface the payload, ask the operator,
    then resume the run via ``Command(resume=...)`` - possibly multiple
    times if the agent triggers another interrupt later in the same turn.
    """
    # First call: pass the user's message.
    payload: Any = {"messages": [input_]}
    seen_messages = 0

    while True:
        result = agent.invoke(payload, config=config)
        # Render any new messages from this slice.
        msgs = result.get("messages", [])
        _render_messages(msgs[seen_messages:])
        seen_messages = len(msgs)

        interrupts = result.get("__interrupt__")
        if not interrupts:
            return

        # interrupts is a tuple of Interrupt objects; each has .value (the payload)
        decision_total: list[Any] = []
        for itr in interrupts:
            decision = _handle_interrupt(itr.value if hasattr(itr, "value") else itr)
            decision_total.append(decision)

        # Resume. If multiple interrupts fired we resume with the last decision
        # (langgraph sends one Command(resume=...) per resume cycle).
        payload = Command(resume=decision_total[-1])


def _handle_interrupt(value: Any) -> Any:
    """Render the interrupt payload and ask the operator y/n."""
    kind = (value or {}).get("kind") if isinstance(value, dict) else None
    if kind == "schema_evolution_approval":
        proposal = value.get("proposal", {})
        body = [
            f"[bold]Activity:[/bold] {value.get('activity_name')}",
            f"[bold]Payloads:[/bold] {value.get('payloads_dir')}",
            "",
            f"[bold]Summary:[/bold] {proposal.get('summary', '')}",
            "",
            "[bold]New labels:[/bold]",
        ]
        for lbl in proposal.get("new_labels", []):
            body.append(
                f"  - [cyan]{lbl['label']}[/cyan] (key={lbl.get('unique_key')}) "
                f"props={lbl.get('properties')}\n    {lbl.get('rationale','')}"
            )
        body.append("\n[bold]New relationships:[/bold]")
        for rel in proposal.get("new_relationships", []):
            body.append(
                f"  - ({rel['from_label']})-[:{rel['type']}]->({rel['to_label']})  "
                f"{rel.get('rationale','')}"
            )
        body.append("\n[bold]Normalization rules:[/bold]")
        for rule in proposal.get("normalization_rules", []):
            body.append(f"  - {rule}")
        console.print(Panel("\n".join(body), title="Schema-evolution approval", border_style="magenta"))
        return "approve" if Confirm.ask("Apply this schema?", default=True) else "decline"

    # Unknown interrupt: just dump it and ask.
    console.print(Panel(json.dumps(value, default=str, indent=2), title="Interrupt", border_style="magenta"))
    return "approve" if Confirm.ask("Proceed?", default=True) else "decline"


def _render_messages(messages: list) -> None:
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                console.print(
                    Panel(
                        Syntax(
                            f"{call['name']}({_pretty_args(call.get('args', {}))})",
                            "python",
                            word_wrap=True,
                        ),
                        title="agent calls skill",
                        border_style="yellow",
                    )
                )
            if msg.content:
                content = _stringify(msg.content)
                if content.strip():
                    console.print(Panel(content.strip(), title="assistant", border_style="green"))
        elif isinstance(msg, ToolMessage):
            text = _stringify(msg.content)
            if msg.name not in _FULL_OUTPUT_TOOLS and len(text) > _TRUNCATE_AT:
                text = text[:_TRUNCATE_AT] + f"\n... ({len(text) - _TRUNCATE_AT} more chars truncated)"
            console.print(Panel(text, title=f"tool result: {msg.name}", border_style="blue"))


def _pretty_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = repr(v)
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c["text"] if isinstance(c, dict) and "text" in c else str(c) for c in content)
    return str(content)
