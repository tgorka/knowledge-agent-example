"""End-to-end demo - shared by the CLI's ``demo`` subcommand and scripts/demo.py."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .facade import KnowledgeAgent

console = Console()

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


PART_1_QUESTIONS = [
    "Which customers spent the most with us so far in 2026?",
    "What jobs are still open at Riverside Apartments, and which are high priority?",
    "What is the average ticket size for commercial vs residential customers, broken down by trade?",
    "Find any customer who hasn't had a job in the last 6 months.",
    "Which technician has completed the most jobs in the last 90 days, and what's their average job value?",
    "Are there any locations where we've had repeat repairs on the same trade within 60 days?",
]

PART_2_QUESTIONS = [
    "Which active permits are expiring in the next 60 days, and what jobs do they belong to?",
    "Which jurisdiction is slowest from permit issuance to first inspection?",
    "Have any locations had more than one permit pulled in the last 12 months? Group by trade.",
    "For completed jobs that required a permit, what's the average gap between permit issue date and job completion?",
    "Show me any permit whose inspection failed at least once before passing.",
    "Are any of our delinquent customers sitting on open permits?",
]


def _pause(auto: bool) -> None:
    if auto:
        return
    try:
        input("\n[press Enter to continue]\n")
    except EOFError:
        return


def _render_answer(ka_question: str, cypher: str, rows: list, answer: str) -> None:
    # Imported here so the module stays import-cheap.
    from .cli import _render_answer as _r
    _r(ka_question, cypher, rows, answer)


def run_demo(ka: KnowledgeAgent | None = None, auto: bool = False) -> None:
    own = ka is None
    ka = ka or KnowledgeAgent()

    try:
        from .cli import _cli_approver, _print_proposal
        from .pipeline import always_approve

        approver = _cli_approver
        if auto:
            # --auto = no stdin prompts. Render the proposal so the operator
            # can still see it scroll past, then auto-approve.
            def _auto_approver(proposal):
                _print_proposal(proposal)
                console.print("[yellow]--auto: auto-approving.[/yellow]")
                return always_approve(proposal)
            approver = _auto_approver

        console.print(Panel.fit("Resetting graph", style="bold magenta"))
        ka.reset()
        _pause(auto)

        console.print(
            Panel.fit("Part 1 - Seeding base schema (Customer / Location / Job)", style="bold magenta")
        )
        ka.seed(DATA_ROOT / "seed")
        _pause(auto)

        console.print(Panel.fit("Part 1 - Agent answers the 6 base questions", style="bold magenta"))
        for q in PART_1_QUESTIONS:
            a = ka.ask(q)
            _render_answer(a.question, a.cypher, a.rows, a.answer)
            _pause(auto)

        console.print(
            Panel.fit("Part 2 - Schema evolution from messy permit payloads", style="bold magenta")
        )
        result = ka.evolve(
            DATA_ROOT / "permits",
            activity_name="permit_evolution_v1",
            approver=approver,
        )
        if not result.applied:
            console.print(
                "[yellow]Operator declined; Part 2 questions will run against the un-evolved graph.[/yellow]"
            )
        _pause(auto)

        console.print(Panel.fit("Part 2 - Agent answers cross-schema questions", style="bold magenta"))
        for q in PART_2_QUESTIONS:
            a = ka.ask(q)
            _render_answer(a.question, a.cypher, a.rows, a.answer)
            _pause(auto)

        console.print(Panel.fit("Demo complete.", style="bold green"))
    finally:
        if own:
            ka.close()
