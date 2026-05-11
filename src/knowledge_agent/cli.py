"""CLI surface. ``knowledge-agent <command>`` after `uv sync`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax

from .facade import KnowledgeAgent
from .pipeline import SchemaProposal

console = Console()

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 1:
        level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def _cli_approver(proposal: SchemaProposal) -> bool:
    """Render the proposal in a rich panel and ask the operator y/n on stdin.

    Used by `knowledge-agent evolve` and the canned `demo` command. The
    chat REPL uses a LangGraph ``interrupt()``-based approver instead.
    """
    _print_proposal(proposal)
    return Confirm.ask("Apply this schema and ingest the payloads?", default=True)


def _print_proposal(proposal: SchemaProposal) -> None:
    body = [f"[bold]Summary:[/bold] {proposal.summary}", "", "[bold]New labels:[/bold]"]
    for lbl in proposal.new_labels:
        body.append(
            f"  - [cyan]{lbl.label}[/cyan] (key={lbl.unique_key}) "
            f"props={lbl.properties}\n    {lbl.rationale}"
        )
    body.append("\n[bold]New relationships:[/bold]")
    for rel in proposal.new_relationships:
        body.append(f"  - ({rel.from_label})-[:{rel.type}]->({rel.to_label})  {rel.rationale}")
    body.append("\n[bold]Normalization rules:[/bold]")
    for rule in proposal.normalization_rules:
        body.append(f"  - {rule}")
    console.print(Panel("\n".join(body), title="Proposed schema evolution"))


def _render_answer(question: str, cypher: str, rows: list, answer: str) -> None:
    console.rule(f"[bold]Q:[/bold] {question}")
    if cypher:
        console.print(Panel(Syntax(cypher, "cypher", word_wrap=True), title="Generated Cypher"))
    if rows:
        keys = sorted({k for r in rows[:10] for k in r.keys()})
        lines = [" | ".join(keys), "-+-".join("-" * len(k) for k in keys)]
        for r in rows[:10]:
            lines.append(" | ".join(_short(r.get(k)) for k in keys))
        if len(rows) > 10:
            lines.append(f"... ({len(rows) - 10} more)")
        console.print(Panel("\n".join(lines), title=f"Result rows ({len(rows)})"))
    console.print(Panel(answer or "(no answer)", title="Answer", border_style="green"))


def _short(v) -> str:
    s = "" if v is None else str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _run_doctor() -> int:
    """One-shot health check for Aura + OpenRouter. Returns the exit code."""
    import neo4j
    import httpx
    from .config import get_settings

    s = get_settings()
    failures = 0

    console.rule("[bold]knowledge-agent doctor[/bold]")

    # ----- Aura ----------------------------------------------------------
    console.print(
        f"[bold]Neo4j[/bold]  uri={s.neo4j_uri}  user={s.neo4j_username!r}  "
        f"db={s.neo4j_database!r}  pass[len]={len(s.neo4j_password)}"
    )
    try:
        drv = neo4j.GraphDatabase.driver(
            s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password)
        )
        try:
            drv.verify_connectivity()
            # verify_connectivity only checks bolt reachability + auth,
            # NOT that the database name is correct. Run a trivial query
            # against the configured database so we surface "database not
            # found" here rather than during the first real call.
            with drv.session(database=s.neo4j_database) as session:
                session.run("RETURN 1 AS ok").single()
            console.print("  [green]ok[/green] connection + auth + database all reachable")
        finally:
            drv.close()
    except neo4j.exceptions.AuthError as e:
        failures += 1
        console.print(
            f"  [red]fail[/red] AuthError: {e.message}\n"
            f"  -> Server is reachable but rejected the password. Re-copy "
            f"NEO4J_PASSWORD from the credentials .txt Aura gave you, or reset "
            f"the password in the Aura console. Username + database must also "
            f"match what's in that file."
        )
    except neo4j.exceptions.ServiceUnavailable as e:
        failures += 1
        console.print(
            f"  [red]fail[/red] ServiceUnavailable: {e}\n"
            f"  -> Could not reach the instance. Check the URI scheme "
            f"(Aura needs neo4j+s://) and whether the instance is paused."
        )
    except neo4j.exceptions.ClientError as e:
        failures += 1
        msg = getattr(e, "message", str(e))
        hint = ""
        if "DatabaseNotFound" in str(e.code or ""):
            hint = (
                f"\n  -> Auth succeeded but database {s.neo4j_database!r} doesn't "
                f"exist on this instance. Depending on Aura edition the database "
                f"name is either 'neo4j' OR the instance id - try the other."
            )
        console.print(f"  [red]fail[/red] {e.code}: {msg}{hint}")
    except Exception as e:
        failures += 1
        console.print(f"  [red]fail[/red] {type(e).__name__}: {e}")

    # ----- OpenRouter -----------------------------------------------------
    console.print(
        f"\n[bold]OpenRouter[/bold]  base={s.openrouter_base_url}  "
        f"model={s.openrouter_model}  key[prefix]={s.openrouter_api_key[:10]}..."
    )
    try:
        r = httpx.get(
            f"{s.openrouter_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {s.openrouter_api_key}"},
            timeout=10.0,
        )
        if r.status_code == 200:
            console.print(
                f"  [green]ok[/green] /models returned 200 ({len(r.json().get('data', []))} models listed)"
            )
        elif r.status_code in (401, 403):
            failures += 1
            console.print(
                f"  [red]fail[/red] {r.status_code} {r.reason_phrase}\n"
                f"  -> OPENROUTER_API_KEY in .env is invalid or revoked. Get a new "
                f"key at https://openrouter.ai/keys."
            )
        else:
            failures += 1
            console.print(
                f"  [yellow]warn[/yellow] /models returned {r.status_code} {r.reason_phrase}"
            )
    except Exception as e:
        failures += 1
        console.print(f"  [red]fail[/red] {type(e).__name__}: {e}")

    console.rule()
    if failures == 0:
        console.print("[green]All checks passed - run `knowledge-agent demo` next.[/green]")
        return 0
    console.print(f"[red]{failures} check(s) failed.[/red]")
    return 1


def run() -> None:
    parser = argparse.ArgumentParser(
        prog="knowledge-agent",
        description="Self-evolving graph agent for FSM data. "
        "Run with no command to print this help.",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("reset", help="Wipe the graph")
    sub.add_parser("seed", help="Import the seed JSON files")
    sub.add_parser("evolve", help="Run schema evolution against data/permits")

    p_ask = sub.add_parser("ask", help="Ask a natural-language question (direct Cypher Q&A)")
    p_ask.add_argument("question", nargs="+")

    sub.add_parser("demo", help="Run the full Part 1 + Part 2 demo end-to-end")
    sub.add_parser("chat", help="Interactive skill-calling agent REPL")
    sub.add_parser("skills", help="List the LangChain skills the chat agent can use")
    sub.add_parser("doctor", help="Check Aura + OpenRouter credentials and connectivity")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    if args.cmd == "skills":
        from .skills import describe_skills

        console.print(describe_skills())
        return

    if args.cmd == "doctor":
        raise SystemExit(_run_doctor())

    from .config import get_settings
    from .repositories.session import Neo4jConnectionError

    _setup_logging(get_settings().agent_verbosity)

    try:
        ka = KnowledgeAgent()
    except Neo4jConnectionError as e:
        console.print(f"[red]Neo4j connection failed.[/red]\n{e}")
        raise SystemExit(2)
    try:
        if args.cmd == "reset":
            ka.reset()
            console.print("[green]Graph wiped.[/green]")
        elif args.cmd == "seed":
            ka.seed(DATA_ROOT / "seed")
            console.print("[green]Seed import complete.[/green]")
        elif args.cmd == "evolve":
            result = ka.evolve(
                DATA_ROOT / "permits",
                activity_name="permit_evolution_v1",
                approver=_cli_approver,
            )
            if result.applied:
                console.print(f"[green]Evolution applied. proposal_id={result.proposal_id}[/green]")
            else:
                console.print("[yellow]Evolution declined; proposal recorded in audit log.[/yellow]")
        elif args.cmd == "ask":
            a = ka.ask(" ".join(args.question))
            _render_answer(a.question, a.cypher, a.rows, a.answer)
        elif args.cmd == "demo":
            from .demo import run_demo

            run_demo(ka)
        elif args.cmd == "chat":
            from .chat import run_chat

            run_chat(ka)
    finally:
        ka.close()
