"""Unit tests for proxy_pointer_rag.tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from akgentic.tool.vector_store.protocol import SearchHit
from proxy_pointer_rag.tools import (
    Answer,
    RagSearchHit,
    Section,
    _find_node_offsets,
    dedup_by_pointer,
    load_section,
    rerank_by_hierarchical_path,
    synthesize,
    vector_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit(
    ref_id: str = "doc1|node1",
    score: float = 0.9,
    text: str = "",
    ref_type: str = "chunk",
) -> SearchHit:
    return SearchHit(ref_type=ref_type, ref_id=ref_id, text=text, score=score)


def _rag_hit(
    doc_id: str = "doc1",
    node_id: str = "node1",
    score: float = 0.9,
    hierarchical_path: list[str] | None = None,
    source_path: str = "doc.md",
    char_start: int = 0,
    char_end: int = 100,
) -> RagSearchHit:
    return RagSearchHit(
        doc_id=doc_id,
        node_id=node_id,
        score=score,
        hierarchical_path=hierarchical_path or ["A"],
        source_path=source_path,
        char_start=char_start,
        char_end=char_end,
    )


# ---------------------------------------------------------------------------
# RagSearchHit.from_search_hit
# ---------------------------------------------------------------------------


class TestRagSearchHitFromSearchHit:
    def test_decodes_ref_id_parts(self) -> None:
        hit = _hit(ref_id="docabc|nodedef")
        rag = RagSearchHit.from_search_hit(hit)
        assert rag.doc_id == "docabc"
        assert rag.node_id == "nodedef"

    def test_ref_id_without_separator(self) -> None:
        hit = _hit(ref_id="onlyid")
        rag = RagSearchHit.from_search_hit(hit)
        assert rag.doc_id == "onlyid"
        assert rag.node_id == ""

    def test_decodes_json_payload_in_text(self) -> None:
        payload = {
            "hierarchical_path": ["A", "B"],
            "source_path": "a/b.md",
            "char_start": 10,
            "char_end": 200,
            "text_preview": "preview text",
        }
        hit = _hit(ref_id="d|n", text=json.dumps(payload))
        rag = RagSearchHit.from_search_hit(hit)
        assert rag.hierarchical_path == ["A", "B"]
        assert rag.source_path == "a/b.md"
        assert rag.char_start == 10
        assert rag.char_end == 200
        assert rag.text_preview == "preview text"

    def test_falls_back_gracefully_on_non_json_text(self) -> None:
        hit = _hit(ref_id="d|n", text="plain text")
        rag = RagSearchHit.from_search_hit(hit)
        assert rag.text_preview == "plain text"
        assert rag.hierarchical_path == []

    def test_score_preserved(self) -> None:
        hit = _hit(ref_id="d|n", score=0.42)
        rag = RagSearchHit.from_search_hit(hit)
        assert rag.score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Section.citation_key
# ---------------------------------------------------------------------------


class TestSectionCitationKey:
    def test_citation_key_format(self) -> None:
        s = Section(doc_id="doc1", node_id="node1", hierarchical_path=["A"], text="text")
        assert s.citation_key == "[doc1#node1]"


# ---------------------------------------------------------------------------
# vector_search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_raises_without_search_fn(self) -> None:
        with pytest.raises(NotImplementedError):
            vector_search("query")

    def test_calls_search_fn_and_returns_rag_hits(self) -> None:
        raw = [_hit(ref_id="d1|n1", score=0.8), _hit(ref_id="d2|n2", score=0.6)]

        def _fn(q: str, k: int, col: str) -> list[SearchHit]:
            return raw

        result = vector_search("query", search_fn=_fn)
        assert len(result) == 2
        assert result[0].doc_id == "d1"
        assert result[1].doc_id == "d2"

    def test_passes_k_and_collection_to_search_fn(self) -> None:
        calls: list[tuple[str, int, str]] = []

        def _fn(q: str, k: int, col: str) -> list[SearchHit]:
            calls.append((q, k, col))
            return []

        vector_search("q", k=50, collection="my_col", search_fn=_fn)
        assert calls[0] == ("q", 50, "my_col")


# ---------------------------------------------------------------------------
# dedup_by_pointer
# ---------------------------------------------------------------------------


class TestDedupByPointer:
    def test_empty_input(self) -> None:
        assert dedup_by_pointer([]) == []

    def test_no_duplicates_unchanged(self) -> None:
        hits = [_rag_hit("d1", "n1"), _rag_hit("d2", "n2")]
        assert dedup_by_pointer(hits) == hits

    def test_deduplicates_same_doc_node(self) -> None:
        hits = [
            _rag_hit("d1", "n1", score=0.9),
            _rag_hit("d1", "n1", score=0.7),
            _rag_hit("d2", "n2", score=0.8),
        ]
        result = dedup_by_pointer(hits)
        assert len(result) == 2
        # First occurrence (highest score) kept
        assert result[0].score == pytest.approx(0.9)

    def test_preserves_order(self) -> None:
        hits = [_rag_hit("d3", "n3"), _rag_hit("d1", "n1"), _rag_hit("d2", "n2")]
        result = dedup_by_pointer(hits)
        assert [h.doc_id for h in result] == ["d3", "d1", "d2"]

    def test_different_doc_same_node_not_deduped(self) -> None:
        hits = [_rag_hit("d1", "n1"), _rag_hit("d2", "n1")]
        result = dedup_by_pointer(hits)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# rerank_by_hierarchical_path
# ---------------------------------------------------------------------------


class TestRerankByHierarchicalPath:
    def test_raises_without_llm_fn(self) -> None:
        with pytest.raises(NotImplementedError):
            rerank_by_hierarchical_path("q", [_rag_hit()])

    def test_empty_hits_returns_empty(self) -> None:
        result = rerank_by_hierarchical_path("q", [], llm_fn=lambda s, u: "[]")
        assert result == []

    def test_returns_top_k_hits(self) -> None:
        hits = [_rag_hit(f"d{i}", f"n{i}") for i in range(10)]
        llm_fn = lambda s, u: json.dumps(list(range(5)))  # noqa: E731
        result = rerank_by_hierarchical_path("q", hits, top_k=5, llm_fn=llm_fn)
        assert len(result) == 5

    def test_respects_llm_ordering(self) -> None:
        hits = [_rag_hit(f"d{i}", f"n{i}") for i in range(4)]
        # LLM says: prefer index 3, 1, 0, 2
        llm_fn = lambda s, u: "[3, 1, 0, 2]"  # noqa: E731
        result = rerank_by_hierarchical_path("q", hits, top_k=4, llm_fn=llm_fn)
        assert [h.doc_id for h in result] == ["d3", "d1", "d0", "d2"]

    def test_falls_back_to_score_order_on_bad_json(self) -> None:
        hits = [_rag_hit(f"d{i}", f"n{i}", score=float(i)) for i in range(3)]
        llm_fn = lambda s, u: "not valid json"  # noqa: E731
        result = rerank_by_hierarchical_path("q", hits, top_k=3, llm_fn=llm_fn)
        assert len(result) == 3

    def test_deduplicates_llm_indices(self) -> None:
        hits = [_rag_hit(f"d{i}", f"n{i}") for i in range(3)]
        llm_fn = lambda s, u: "[0, 0, 1]"  # noqa: E731
        result = rerank_by_hierarchical_path("q", hits, top_k=3, llm_fn=llm_fn)
        doc_ids = [h.doc_id for h in result]
        assert len(doc_ids) == len(set(doc_ids))

    def test_clamps_to_available_hits(self) -> None:
        hits = [_rag_hit("d1", "n1")]
        llm_fn = lambda s, u: "[0]"  # noqa: E731
        result = rerank_by_hierarchical_path("q", hits, top_k=5, llm_fn=llm_fn)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _find_node_offsets
# ---------------------------------------------------------------------------


class TestFindNodeOffsets:
    def test_finds_node_in_flat_list(self) -> None:
        nodes = [
            {"node_id": "n1", "char_start": 0, "char_end": 50, "children": []},
            {"node_id": "n2", "char_start": 50, "char_end": 100, "children": []},
        ]
        assert _find_node_offsets(nodes, "n2") == (50, 100)

    def test_finds_nested_node(self) -> None:
        nodes = [
            {
                "node_id": "n1",
                "char_start": 0,
                "char_end": 100,
                "children": [
                    {"node_id": "n2", "char_start": 10, "char_end": 80, "children": []}
                ],
            }
        ]
        assert _find_node_offsets(nodes, "n2") == (10, 80)

    def test_returns_minus_one_when_not_found(self) -> None:
        nodes: list[dict[str, object]] = []
        assert _find_node_offsets(nodes, "missing") == (-1, -1)


# ---------------------------------------------------------------------------
# load_section
# ---------------------------------------------------------------------------


class TestLoadSection:
    def test_loads_from_hits_fast_path(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("Hello World Section Content")
        hit = _rag_hit(
            doc_id="doc1",
            node_id="node1",
            source_path="doc.md",
            char_start=6,
            char_end=11,
        )
        result = load_section("doc1", "node1", md_root=tmp_path, hits=[hit])
        assert result == "World"

    def test_raises_file_not_found_when_no_skeleton(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_section("nonexistent_doc", "node1", md_root=tmp_path)


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_raises_without_llm_fn(self) -> None:
        with pytest.raises(NotImplementedError):
            synthesize("q", [])

    def test_returns_answer_with_text(self) -> None:
        sections = [
            Section(doc_id="d1", node_id="n1", hierarchical_path=["A"], text="content")
        ]
        llm_fn = lambda s, u: "Answer text [d1#n1] is here."  # noqa: E731
        result = synthesize("query", sections, llm_fn=llm_fn)
        assert isinstance(result, Answer)
        assert "Answer text" in result.text

    def test_extracts_citations(self) -> None:
        sections = [
            Section(doc_id="doc1", node_id="node1", hierarchical_path=["A"], text="text")
        ]
        llm_fn = lambda s, u: "See [doc1#node1] and also [doc2#node2]."  # noqa: E731
        result = synthesize("q", sections, llm_fn=llm_fn)
        assert "[doc1#node1]" in result.citations
        assert "[doc2#node2]" in result.citations

    def test_deduplicates_citations(self) -> None:
        sections = [
            Section(doc_id="d1", node_id="n1", hierarchical_path=["A"], text="t")
        ]
        llm_fn = lambda s, u: "[d1#n1] repeated [d1#n1] again."  # noqa: E731
        result = synthesize("q", sections, llm_fn=llm_fn)
        assert result.citations.count("[d1#n1]") == 1

    def test_empty_sections_still_calls_llm(self) -> None:
        called: list[bool] = []
        def llm_fn(s: str, u: str) -> str:
            called.append(True)
            return "No relevant sections found."
        result = synthesize("q", [], llm_fn=llm_fn)
        assert called
        assert result.citations == []
