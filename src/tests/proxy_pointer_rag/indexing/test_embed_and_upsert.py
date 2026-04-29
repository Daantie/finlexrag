"""Unit tests for proxy_pointer_rag.indexing.embed_and_upsert."""

from __future__ import annotations

import hashlib

import pytest

from proxy_pointer_rag.indexing.embed_and_upsert import _point_id, _upsert_with_payload
from proxy_pointer_rag.indexing.models import QdrantPayload


# ---------------------------------------------------------------------------
# _point_id
# ---------------------------------------------------------------------------


class TestPointId:
    def test_returns_uuid_formatted_string(self) -> None:
        pid = _point_id("doc1", "node1", 0)
        parts = pid.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_deterministic(self) -> None:
        assert _point_id("doc1", "node1", 0) == _point_id("doc1", "node1", 0)

    def test_different_inputs_produce_different_ids(self) -> None:
        assert _point_id("doc1", "node1", 0) != _point_id("doc1", "node1", 1)
        assert _point_id("doc1", "node1", 0) != _point_id("doc2", "node1", 0)
        assert _point_id("doc1", "node1", 0) != _point_id("doc1", "node2", 0)

    def test_derived_from_sha1(self) -> None:
        raw = "doc1|node1|0"
        digest = hashlib.sha1(raw.encode()).hexdigest()
        expected = f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
        assert _point_id("doc1", "node1", 0) == expected

    def test_chunk_idx_zero_and_nonzero_differ(self) -> None:
        assert _point_id("d", "n", 0) != _point_id("d", "n", 99)


# ---------------------------------------------------------------------------
# _upsert_with_payload
# ---------------------------------------------------------------------------


class TestUpsertWithPayload:
    def test_upserts_points_with_rich_payload(self) -> None:
        """_upsert_with_payload stores QdrantPayload fields in Qdrant."""
        from akgentic.tool.vector import VectorEntry
        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.qdrant import QdrantBackend

        backend = QdrantBackend()
        backend.create_collection("test_col", CollectionConfig(backend="qdrant", dimension=3))

        pid = _point_id("doc1", "node1", 0)
        entry = VectorEntry(ref_type="chunk", ref_id=pid, text="hello", vector=[1.0, 0.0, 0.0])
        payload = QdrantPayload(
            doc_id="doc1",
            node_id="node1",
            hierarchical_path=["A", "B"],
            source_path="a/b.md",
            char_start=0,
            char_end=100,
            text_preview="hello",
        )

        _upsert_with_payload(backend, "test_col", [entry], [payload])

        result = backend.search("test_col", [1.0, 0.0, 0.0], top_k=1)
        assert len(result.hits) == 1

    def test_upsert_is_idempotent(self) -> None:
        """Calling _upsert_with_payload twice with same data does not duplicate."""
        from akgentic.tool.vector import VectorEntry
        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.qdrant import QdrantBackend

        backend = QdrantBackend()
        backend.create_collection("test_col", CollectionConfig(backend="qdrant", dimension=3))

        pid = _point_id("doc1", "node1", 0)
        entry = VectorEntry(ref_type="chunk", ref_id=pid, text="hello", vector=[1.0, 0.0, 0.0])
        payload = QdrantPayload(
            doc_id="doc1",
            node_id="node1",
            hierarchical_path=["A"],
            source_path="a.md",
            char_start=0,
            char_end=50,
            text_preview="hello",
        )

        _upsert_with_payload(backend, "test_col", [entry], [payload])
        _upsert_with_payload(backend, "test_col", [entry], [payload])

        result = backend.search("test_col", [1.0, 0.0, 0.0], top_k=10)
        assert len(result.hits) == 1
