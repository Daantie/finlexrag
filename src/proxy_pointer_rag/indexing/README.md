# Proxy-Pointer RAG — Indexing Pipeline

The `proxy_pointer_rag.indexing` module provides an offline pipeline to convert source documents into a hierarchical, section-aware RAG index stored in Qdrant.

## Overview

The pipeline follows a 3-stage process:

1.  **Parse**: Converts mixed source documents (`.pdf`, `.docx`, `.pptx`, etc.) into clean Markdown files using [docling](https://github.com/DS4SD/docling).
2.  **Skeleton**: Deterministically parses the Markdown heading structure to create a hierarchical `skeleton.json` for each document.
3.  **Index**: 
    - Applies an LLM-based **noise filter** to drop non-substantive sections (TOCs, abbreviations, boilerplate).
    - **Chunks** surviving sections while respecting heading boundaries and token budgets.
    - **Embeds** chunks via OpenAI.
    - **Upserts** vectors and rich metadata into **Qdrant** with deterministic point IDs for idempotency.

## Prerequisites

### API Keys
The pipeline requires an `OPENAI_API_KEY` environment variable for:
- Noise filtering (default model: `gpt-4.1-mini`).
- Embeddings (model: `text-embedding-3-small`).

### Vector Store
A Qdrant instance must be reachable. You can start the local development stack via:
```bash
docker-compose up -d qdrant
```
By default, the indexer connects to `http://localhost:6333`.

### Dependencies
Ensure the workspace is synced with all extras:
```bash
uv sync --all-packages --all-extras
```
*Note: `docling` is a large dependency and is included via the `docs_docling` extra.*

## Usage

The pipeline is invoked via `python -m proxy_pointer_rag.indexing`.

### 1. Parse Stage
Convert source files to Markdown.
```bash
export PYTHONPATH=src
python -m proxy_pointer_rag.indexing parse \
    --input-dir data/source_docs \
    --output-dir output/markdown
```

### 2. Skeleton Stage
Generate hierarchical heading trees.
```bash
python -m proxy_pointer_rag.indexing skeleton \
    --input-dir output/markdown \
    --output-dir output/skeletons
```

### 3. Index Stage
Noise-filter, chunk, embed, and upsert to Qdrant.
```bash
python -m proxy_pointer_rag.indexing index \
    --md-dir output/markdown \
    --skeleton-dir output/skeletons \
    --collection my_collection_name
```

## Data Models

The pipeline uses Pydantic models defined in `models.py`:
- `SkeletonNode`: Represents a heading/section in the hierarchy.
- `Chunk`: A text segment within a section.
- `QdrantPayload`: The metadata stored alongside vectors, including `doc_id`, `node_id`, `hierarchical_path`, and character offsets for on-demand section loading.

## Architecture & Idempotency

- **Stable IDs**: `doc_id` is derived from the relative file path. `node_id` is derived from `doc_id` and the hierarchical path.
- **Deterministic Point IDs**: Qdrant point IDs are `sha1(doc_id|node_id|chunk_idx)`, ensuring that re-running the indexer on the same files overwrites existing points rather than duplicating them.
- **Layering**: The pipeline reuses `EmbeddingService` and `VectorStoreActor` (Qdrant backend) from `akgentic-tool`.
