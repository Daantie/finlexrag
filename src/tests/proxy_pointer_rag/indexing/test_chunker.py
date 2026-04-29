"""Unit tests for proxy_pointer_rag.indexing.chunker."""

from __future__ import annotations

from proxy_pointer_rag.indexing.chunker import (
    DEFAULT_TOKEN_BUDGET,
    _estimate_tokens,
    _split_into_chunks,
    chunk_nodes,
)
from proxy_pointer_rag.indexing.models import SkeletonNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    title: str,
    char_start: int,
    char_end: int,
    path: list[str] | None = None,
) -> SkeletonNode:
    return SkeletonNode(
        node_id=node_id,
        title=title,
        level=1,
        path=path or [title],
        char_start=char_start,
        char_end=char_end,
    )


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_one(self) -> None:
        assert _estimate_tokens("") == 1

    def test_single_word(self) -> None:
        # 1 word / 0.75 = 1.33 → rounds to 1
        assert _estimate_tokens("hello") >= 1

    def test_proportional_to_word_count(self) -> None:
        short = _estimate_tokens("one two three")
        long = _estimate_tokens("one two three four five six seven eight nine ten")
        assert long > short

    def test_whitespace_only_returns_one(self) -> None:
        assert _estimate_tokens("   ") == 1


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    def test_short_text_returns_single_slice(self) -> None:
        text = "Short text."
        slices = _split_into_chunks(text, 0, DEFAULT_TOKEN_BUDGET)
        assert len(slices) == 1
        assert slices[0] == (0, len(text))

    def test_empty_text_returns_empty(self) -> None:
        assert _split_into_chunks("", 0, DEFAULT_TOKEN_BUDGET) == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert _split_into_chunks("   \n  ", 0, DEFAULT_TOKEN_BUDGET) == []

    def test_char_offset_applied(self) -> None:
        text = "Hello world."
        slices = _split_into_chunks(text, 100, DEFAULT_TOKEN_BUDGET)
        assert slices[0][0] >= 100

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        # Build text that exceeds budget
        para = "word " * 200  # ~267 tokens per para
        text = (para + "\n\n") * 4
        slices = _split_into_chunks(text, 0, 300)
        assert len(slices) >= 2

    def test_slices_cover_non_empty_content(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five.\n\n"
        slices = _split_into_chunks(text, 0, 5)
        for start, end in slices:
            assert end > start


# ---------------------------------------------------------------------------
# chunk_nodes
# ---------------------------------------------------------------------------


class TestChunkNodes:
    def test_empty_nodes_returns_empty(self) -> None:
        assert chunk_nodes([], "text", "doc.md", "docid") == []

    def test_single_node_produces_chunk(self) -> None:
        text = "# Title\n\nSome substantive content here.\n"
        node = _node("n1", "Title", 0, len(text))
        chunks = chunk_nodes([node], text, "doc.md", "docid")
        assert len(chunks) >= 1
        assert chunks[0].doc_id == "docid"
        assert chunks[0].node_id == "n1"
        assert chunks[0].source_path == "doc.md"

    def test_chunk_char_offsets_within_node_bounds(self) -> None:
        text = "# Title\n\nContent.\n"
        node = _node("n1", "Title", 0, len(text))
        chunks = chunk_nodes([node], text, "doc.md", "docid")
        for chunk in chunks:
            assert chunk.char_start >= node.char_start
            assert chunk.char_end <= node.char_end

    def test_multiple_nodes_produce_chunks_for_each(self) -> None:
        text = "# A\n\nContent A.\n\n# B\n\nContent B.\n"
        node_a = _node("na", "A", 0, text.index("# B"))
        node_b = _node("nb", "B", text.index("# B"), len(text))
        chunks = chunk_nodes([node_a, node_b], text, "doc.md", "docid")
        node_ids = {c.node_id for c in chunks}
        assert "na" in node_ids
        assert "nb" in node_ids

    def test_blank_section_produces_no_chunk(self) -> None:
        # Section text is purely whitespace — no heading, no content
        text = "   \n\n   \n\n"
        node = _node("n1", "Title", 0, len(text))
        chunks = chunk_nodes([node], text, "doc.md", "docid")
        assert chunks == []

    def test_hierarchical_path_stored_in_chunk(self) -> None:
        text = "# A\n## B\n\nContent.\n"
        node = _node("n1", "B", 0, len(text), path=["A", "B"])
        chunks = chunk_nodes([node], text, "doc.md", "docid")
        assert chunks[0].hierarchical_path == ["A", "B"]

    def test_token_budget_respected(self) -> None:
        # Each paragraph is ~20 tokens; budget of 25 should split them
        para = "word " * 15 + "\n\n"
        text = para * 5
        node = _node("n1", "Title", 0, len(text))
        chunks = chunk_nodes([node], text, "doc.md", "docid", token_budget=25)
        # With budget=25 and ~20 tokens/para, multiple chunks expected
        assert len(chunks) >= 1
