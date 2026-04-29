"""Proxy-Pointer RAG — embed and upsert stage.

Orchestrates the full ``index`` pipeline for each document:

1. Load skeleton.json and .md file.
2. Flatten skeleton nodes and run the LLM noise filter.
3. Chunk the surviving nodes (section-aware, token-budget).
4. Embed each chunk via ``EmbeddingService``.
5. Upsert into Qdrant with deterministic point IDs (sha1(doc_id|node_id|chunk_idx)).

Point IDs are deterministic so re-running the stage on the same input produces
the same Qdrant state (idempotency via upsert).

See docs/requirements.md → Functional requirements → Indexing → ``index`` steps 3–4.
See docs/requirements.md → Non-functional → Idempotency.
See docs/requirements.md → Non-functional → Reuses existing infra.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from akgentic.tool.vector import VectorEntry

from proxy_pointer_rag.indexing.chunker import DEFAULT_TOKEN_BUDGET, chunk_nodes
from proxy_pointer_rag.indexing.models import QdrantPayload, SkeletonNode
from proxy_pointer_rag.indexing.noise_filter import filter_nodes_sync, flatten_nodes
from proxy_pointer_rag.indexing.parse import doc_id_for

logger = logging.getLogger(__name__)

# Preview length stored in QdrantPayload.text_preview
_PREVIEW_LEN = 256

# Embedding model used for indexing
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_PROVIDER = "openai"


def _point_id(doc_id: str, node_id: str, chunk_idx: int) -> str:
    """Return a deterministic Qdrant point ID as a UUID-formatted sha1 digest.

    The ID is derived from ``sha1(doc_id|node_id|chunk_idx)`` so that
    re-indexing the same content always produces the same point ID, enabling
    idempotent upserts.

    Args:
        doc_id: Stable document identifier.
        node_id: Stable node identifier within the document.
        chunk_idx: Zero-based chunk index within the node.

    Returns:
        A UUID-formatted string (8-4-4-4-12 hex) derived from the sha1 digest.
    """
    raw = f"{doc_id}|{node_id}|{chunk_idx}"
    digest = hashlib.sha1(raw.encode()).hexdigest()
    # Format as UUID: 8-4-4-4-12
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _index_document(
    md_path: Path,
    skeleton_path: Path,
    md_root: Path,
    collection: str,
    backend: object,
    model: str,
    token_budget: int,
) -> int:
    """Index a single document: filter → chunk → embed → upsert.

    Args:
        md_path: Absolute path to the .md file.
        skeleton_path: Absolute path to the corresponding skeleton.json.
        md_root: Root directory of .md files (used to derive relative path).
        collection: Qdrant collection name.
        backend: ``QdrantBackend`` instance.
        model: LLM model for noise filtering.
        token_budget: Token budget per chunk.

    Returns:
        Number of points upserted.
    """
    from akgentic.tool.vector import EmbeddingService
    from akgentic.tool.vector import VectorEntry
    from akgentic.tool.vector_store.qdrant import QdrantBackend

    assert isinstance(backend, QdrantBackend)

    rel = md_path.relative_to(md_root)
    source_path = str(rel)
    doc_id = doc_id_for(source_path)

    md_text = md_path.read_text(encoding="utf-8")
    raw = json.loads(skeleton_path.read_text(encoding="utf-8"))
    roots = [SkeletonNode.model_validate(n) for n in raw]
    flat_nodes = flatten_nodes(roots)

    if not flat_nodes:
        logger.info("index: no nodes in skeleton for %s — skipping", rel)
        return 0

    # Step 1: noise filter
    filtered_nodes = filter_nodes_sync(flat_nodes, md_text, model=model)
    if not filtered_nodes:
        logger.info("index: all nodes filtered out for %s — skipping", rel)
        return 0

    # Step 2: chunk
    chunks = chunk_nodes(filtered_nodes, md_text, source_path, doc_id, token_budget)
    if not chunks:
        logger.info("index: no chunks produced for %s — skipping", rel)
        return 0

    # Step 3: embed
    embedding_svc = EmbeddingService(model=_EMBEDDING_MODEL, provider="openai")
    texts = [c.text for c in chunks]
    vectors: list[list[float]] = embedding_svc.embed(texts)

    # Step 4: build VectorEntry list with deterministic IDs and upsert
    entries: list[VectorEntry] = []
    payloads: list[QdrantPayload] = []

    for chunk_idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        pid = _point_id(doc_id, chunk.node_id, chunk_idx)
        payload = QdrantPayload(
            doc_id=chunk.doc_id,
            node_id=chunk.node_id,
            hierarchical_path=chunk.hierarchical_path,
            source_path=chunk.source_path,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            text_preview=chunk.text[:_PREVIEW_LEN],
        )
        payloads.append(payload)
        entries.append(
            VectorEntry(
                ref_type="chunk",
                ref_id=pid,
                text=chunk.text,
                vector=vector,
            )
        )

    # Upsert via QdrantBackend — uses ref_id as point ID (uuid5 internally)
    # We override with our deterministic IDs by upserting directly via the client
    _upsert_with_payload(backend, collection, entries, payloads)

    logger.info(
        "index: upserted %d point(s) for %s  [doc_id=%s]",
        len(entries),
        rel,
        doc_id,
    )
    return len(entries)


def _upsert_with_payload(
    backend: object,
    collection: str,
    entries: list["VectorEntry"],
    payloads: list[QdrantPayload],
) -> None:
    """Upsert points into Qdrant with full ``QdrantPayload`` metadata.

    Uses the Qdrant client directly (via ``backend._client``) so we can store
    the rich ``QdrantPayload`` fields alongside the vector, rather than the
    minimal ``{ref_type, ref_id, text}`` payload that ``QdrantBackend.add``
    stores.

    Args:
        backend: ``QdrantBackend`` instance.
        collection: Target collection name.
        entries: ``VectorEntry`` list (ref_id used as point ID).
        payloads: Corresponding ``QdrantPayload`` list.
    """
    from qdrant_client.models import PointStruct

    points = []
    for entry, payload in zip(entries, payloads):
        points.append(
            PointStruct(
                id=entry.ref_id,
                vector=entry.vector,
                payload=payload.model_dump(),
            )
        )

    # Access the underlying qdrant client directly for rich payload upsert
    client = getattr(backend, "_client")
    client.upsert(collection_name=collection, points=points)


def run_index(
    md_dir: str,
    skeleton_dir: str,
    collection: str,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    vector_dimension: int = 1536,
) -> None:
    """Run the full index stage: noise-filter, chunk, embed, and upsert.

    For each .md file in *md_dir* that has a corresponding skeleton.json in
    *skeleton_dir*, the function:

    1. Flattens the skeleton tree and runs the LLM noise filter.
    2. Chunks surviving nodes within *token_budget*.
    3. Embeds chunks via ``EmbeddingService`` (``text-embedding-3-small``).
    4. Upserts into Qdrant with deterministic point IDs.

    The Qdrant collection is created if it does not already exist.

    Args:
        md_dir: Directory containing .md files (output of ``parse``).
        skeleton_dir: Directory containing skeleton.json files (output of ``skeleton``).
        collection: Qdrant collection name to upsert into.
        qdrant_url: Qdrant server URL (default: ``http://localhost:6333``).
        qdrant_api_key: Optional Qdrant API key.
        model: LLM model for noise filtering (default: ``gpt-4.1-mini``).
        token_budget: Maximum tokens per chunk (default: 512).
        vector_dimension: Embedding vector dimension (default: 1536 for
            ``text-embedding-3-small``).
    """
    from akgentic.tool.vector_store.protocol import CollectionConfig
    from akgentic.tool.vector_store.qdrant import QdrantBackend

    md_root = Path(md_dir).resolve()
    skel_root = Path(skeleton_dir).resolve()

    if not md_root.is_dir():
        raise ValueError(f"md-dir does not exist or is not a directory: {md_root}")
    if not skel_root.is_dir():
        raise ValueError(f"skeleton-dir does not exist or is not a directory: {skel_root}")

    backend = QdrantBackend(url=qdrant_url, api_key=qdrant_api_key)

    # Ensure collection exists with HNSW config for k=200 queries (task 3.7)
    collection_cfg = CollectionConfig(
        backend="qdrant",
        dimension=vector_dimension,
    )
    backend.create_collection(collection, collection_cfg)

    md_files = sorted(p for p in md_root.rglob("*.md") if p.is_file())
    if not md_files:
        logger.warning("No .md files found under %s", md_root)
        return

    logger.info("Found %d .md file(s) to index", len(md_files))

    total_points = 0
    skipped = errors = 0

    for md_path in md_files:
        rel = md_path.relative_to(md_root)
        skeleton_path = skel_root / rel.parent / (rel.stem + ".skeleton.json")

        if not skeleton_path.exists():
            logger.warning(
                "index: no skeleton.json for %s (expected %s) — skipping",
                rel,
                skeleton_path,
            )
            skipped += 1
            continue

        try:
            n = _index_document(
                md_path=md_path,
                skeleton_path=skeleton_path,
                md_root=md_root,
                collection=collection,
                backend=backend,
                model=model,
                token_budget=token_budget,
            )
            total_points += n
        except Exception:
            logger.exception("index: failed to index %s", md_path)
            errors += 1

    logger.info(
        "index complete — total_points=%d  skipped=%d  errors=%d",
        total_points,
        skipped,
        errors,
    )
