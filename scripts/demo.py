"""Convenience entry point: ``uv run python scripts/demo.py``."""

from __future__ import annotations

import argparse

from knowledge_agent.demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FSM knowledge-agent demo.")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip the 'press Enter' pauses between sections.",
    )
    args = parser.parse_args()
    run_demo(auto=args.auto)


if __name__ == "__main__":
    main()
