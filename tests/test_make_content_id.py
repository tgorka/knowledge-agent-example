"""``make_content_id`` is a pure function but it's the identity anchor for
every node in the graph. If it ever becomes non-deterministic or starts
producing collisions across namespaces we get duplicated/merged nodes.
"""

from __future__ import annotations

from knowledge_agent.prov import make_content_id


def test_deterministic_for_same_inputs() -> None:
    a = make_content_id("Customer", "c1")
    b = make_content_id("Customer", "c1")
    assert a == b


def test_namespace_isolates() -> None:
    """Same natural key under two namespaces -> two different ids."""
    a = make_content_id("Customer", "c1")
    b = make_content_id("Location", "c1")
    assert a != b


def test_different_keys_different_ids() -> None:
    a = make_content_id("Customer", "c1")
    b = make_content_id("Customer", "c2")
    assert a != b


def test_format_is_q_plus_12_hex() -> None:
    cid = make_content_id("Customer", "c1")
    assert cid.startswith("Q")
    assert len(cid) == 13
    int(cid[1:], 16)
