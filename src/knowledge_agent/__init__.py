"""Self-evolving graph agent for FSM data (Neo4j + LangChain + OpenRouter)."""

import warnings as _warnings

# LangChain's import chain calls ``surface_langchain_deprecation_warnings``
# which inserts a "default" filter for ``LangChainPendingDeprecationWarning``
# *after* anything we set, so a noisy ``allowed_objects`` warning from
# ``langchain_core.load`` fires on every startup. ``catch_warnings(record=True)``
# buffers warnings into a list instead of printing them and is immune to
# later filter-list mutations during the import. Normal warning behaviour
# resumes once the ``with`` block exits.
with _warnings.catch_warnings(record=True):
    _warnings.simplefilter("always")
    from .facade import KnowledgeAgent  # noqa: F401, E402

__all__ = ["KnowledgeAgent", "main"]


def main() -> None:
    from .cli import run

    run()
