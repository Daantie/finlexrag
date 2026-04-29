"""Unit tests for proxy_pointer_rag.indexing.models."""

import pytest

from proxy_pointer_rag.indexing.models import Chunk, QdrantPayload, SkeletonNode


class TestSkeletonNode:
    def test_basic_construction(self) -> None:
        node = SkeletonNode(
            node_id="abc123",
            title="Introduction",
            level=1,
            path=["Introduction"],
            char_start=0,
            char_end=100,
        )
        assert node.title == "Introduction"
        assert node.level == 1
        assert node.children == []

    def test_nested_children(self) -> None:
        child = SkeletonNode(
            node_id="child1",
            title="Sub",
            level=2,
            path=["Introduction", "Sub"],
            char_start=10,
            char_end=50,
        )
        parent = SkeletonNode(
            node_id="parent1",
            title="Introduction",
            level=1,
            path=["Introduction"],
            char_start=0,
            char_end=100,
            children=[child],
        )
        assert len(parent.children) == 1
        assert parent.children[0].title == "Sub"

    def test_make_node_id_stable(self) -> None:
        nid1 = SkeletonNode.make_node_id("docabc", ["Intro", "Background"])
        nid2 = SkeletonNode.make_node_id("docabc", ["Intro", "Background"])
        assert nid1 == nid2
        assert len(nid1) == 12

    def test_make_node_id_differs_by_doc(self) -> None:
        nid1 = SkeletonNode.make_node_id("doc1", ["Intro"])
        nid2 = SkeletonNode.make_node_id("doc2", ["Intro"])
        assert nid1 != nid2

    def test_make_node_id_differs_by_path(self) -> None:
        nid1 = SkeletonNode.make_node_id("doc1", ["Intro"])
        nid2 = SkeletonNode.make_node_id("doc1", ["Intro", "Sub"])
        assert nid1 != nid2

    def test_serialisation_round_trip(self) -> None:
        node = SkeletonNode(
            node_id="abc",
            title="T",
            level=2,
            path=["T"],
            char_start=5,
            char_end=20,
        )
        data = node.model_dump()
        restored = SkeletonNode.model_validate(data)
        assert restored == node


class TestChunk:
    def test_basic_construction(self) -> None:
        chunk = Chunk(
            doc_id="d1",
            node_id="n1",
            hierarchical_path=["Intro", "Background"],
            source_path="docs/intro.md",
            char_start=0,
            char_end=200,
            text="Some text here.",
        )
        assert chunk.doc_id == "d1"
        assert chunk.text == "Some text here."

    def test_serialisation_round_trip(self) -> None:
        chunk = Chunk(
            doc_id="d1",
            node_id="n1",
            hierarchical_path=["A"],
            source_path="a.md",
            char_start=0,
            char_end=10,
            text="hello",
        )
        assert Chunk.model_validate(chunk.model_dump()) == chunk


class TestQdrantPayload:
    def test_basic_construction(self) -> None:
        payload = QdrantPayload(
            doc_id="d1",
            node_id="n1",
            hierarchical_path=["Intro"],
            source_path="intro.md",
            char_start=0,
            char_end=256,
            text_preview="First 256 chars…",
        )
        assert payload.text_preview == "First 256 chars…"

    def test_serialisation_round_trip(self) -> None:
        payload = QdrantPayload(
            doc_id="d1",
            node_id="n1",
            hierarchical_path=["A", "B"],
            source_path="a/b.md",
            char_start=10,
            char_end=300,
            text_preview="preview",
        )
        assert QdrantPayload.model_validate(payload.model_dump()) == payload
