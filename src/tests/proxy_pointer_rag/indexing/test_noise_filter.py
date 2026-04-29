"""Unit tests for proxy_pointer_rag.indexing.noise_filter."""

from __future__ import annotations

import pytest

from proxy_pointer_rag.indexing.models import SkeletonNode
from proxy_pointer_rag.indexing.noise_filter import flatten_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    title: str,
    level: int = 1,
    path: list[str] | None = None,
    char_start: int = 0,
    char_end: int = 100,
    children: list[SkeletonNode] | None = None,
) -> SkeletonNode:
    return SkeletonNode(
        node_id=node_id,
        title=title,
        level=level,
        path=path or [title],
        char_start=char_start,
        char_end=char_end,
        children=children or [],
    )


# ---------------------------------------------------------------------------
# flatten_nodes
# ---------------------------------------------------------------------------


class TestFlattenNodes:
    def test_empty_list(self) -> None:
        assert flatten_nodes([]) == []

    def test_single_node_no_children(self) -> None:
        n = _node("n1", "A")
        result = flatten_nodes([n])
        assert result == [n]

    def test_flat_list_of_siblings(self) -> None:
        nodes = [_node("n1", "A"), _node("n2", "B"), _node("n3", "C")]
        result = flatten_nodes(nodes)
        assert result == nodes

    def test_nested_children_depth_first(self) -> None:
        child = _node("c1", "Child", level=2, path=["Parent", "Child"])
        parent = _node("p1", "Parent", children=[child])
        result = flatten_nodes([parent])
        assert len(result) == 2
        assert result[0].node_id == "p1"
        assert result[1].node_id == "c1"

    def test_deeply_nested(self) -> None:
        grandchild = _node("gc", "GC", level=3, path=["A", "B", "GC"])
        child = _node("c", "B", level=2, path=["A", "B"], children=[grandchild])
        root = _node("r", "A", level=1, path=["A"], children=[child])
        result = flatten_nodes([root])
        assert [n.node_id for n in result] == ["r", "c", "gc"]

    def test_multiple_roots_with_children(self) -> None:
        child1 = _node("c1", "C1", level=2, path=["A", "C1"])
        child2 = _node("c2", "C2", level=2, path=["B", "C2"])
        root1 = _node("r1", "A", children=[child1])
        root2 = _node("r2", "B", children=[child2])
        result = flatten_nodes([root1, root2])
        assert [n.node_id for n in result] == ["r1", "c1", "r2", "c2"]


# ---------------------------------------------------------------------------
# filter_nodes (async) — tested via filter_nodes_sync with mocked LLM
# ---------------------------------------------------------------------------


class TestFilterNodesSync:
    def test_keeps_all_when_llm_returns_keep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When LLM always returns 'keep', all nodes are preserved."""
        import proxy_pointer_rag.indexing.noise_filter as mod

        async def _fake_classify(agent: object, node: SkeletonNode, snippet: str) -> bool:
            return True

        monkeypatch.setattr(mod, "_classify_section", _fake_classify)

        # Patch the imports inside filter_nodes so no real LLM is needed
        import types

        fake_agent = types.SimpleNamespace(run=None)

        async def _fake_filter(
            nodes: list[SkeletonNode],
            md_text: str,
            model: str = "gpt-4.1-mini",
        ) -> list[SkeletonNode]:
            kept = []
            for node in nodes:
                snippet = md_text[node.char_start : node.char_end]
                if await _fake_classify(fake_agent, node, snippet):
                    kept.append(node)
            return kept

        monkeypatch.setattr(mod, "filter_nodes", _fake_filter)

        nodes = [_node("n1", "A"), _node("n2", "B")]
        result = mod.filter_nodes_sync(nodes, "some text")
        assert result == nodes

    def test_drops_all_when_llm_returns_drop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When LLM always returns 'drop', all nodes are removed."""
        import proxy_pointer_rag.indexing.noise_filter as mod

        async def _fake_filter(
            nodes: list[SkeletonNode],
            md_text: str,
            model: str = "gpt-4.1-mini",
        ) -> list[SkeletonNode]:
            return []

        monkeypatch.setattr(mod, "filter_nodes", _fake_filter)

        nodes = [_node("n1", "A"), _node("n2", "B")]
        result = mod.filter_nodes_sync(nodes, "some text")
        assert result == []

    def test_empty_input_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import proxy_pointer_rag.indexing.noise_filter as mod

        async def _fake_filter(
            nodes: list[SkeletonNode],
            md_text: str,
            model: str = "gpt-4.1-mini",
        ) -> list[SkeletonNode]:
            return []

        monkeypatch.setattr(mod, "filter_nodes", _fake_filter)
        result = mod.filter_nodes_sync([], "")
        assert result == []


# ---------------------------------------------------------------------------
# _classify_section — defaults to keep on LLM failure
# ---------------------------------------------------------------------------


class TestClassifySection:
    def test_defaults_to_keep_on_exception(self) -> None:
        """_classify_section returns True (keep) when the LLM call raises."""
        import asyncio

        from proxy_pointer_rag.indexing.noise_filter import _classify_section

        class _FailingAgent:
            async def run(self, prompt: str) -> object:
                raise RuntimeError("LLM unavailable")

        node = _node("n1", "A")
        result = asyncio.run(_classify_section(_FailingAgent(), node, "snippet"))
        assert result is True

    def test_returns_false_for_drop_response(self) -> None:
        import asyncio
        import types

        from proxy_pointer_rag.indexing.noise_filter import _classify_section

        class _DropAgent:
            async def run(self, prompt: str) -> object:
                return types.SimpleNamespace(output="drop")

        node = _node("n1", "TOC")
        result = asyncio.run(_classify_section(_DropAgent(), node, "1. Introduction\n2. Scope"))
        assert result is False

    def test_returns_true_for_keep_response(self) -> None:
        import asyncio
        import types

        from proxy_pointer_rag.indexing.noise_filter import _classify_section

        class _KeepAgent:
            async def run(self, prompt: str) -> object:
                return types.SimpleNamespace(output="keep")

        node = _node("n1", "Scope")
        result = asyncio.run(_classify_section(_KeepAgent(), node, "This regulation applies to..."))
        assert result is True
