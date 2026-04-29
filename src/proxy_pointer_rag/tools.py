"""Proxy-Pointer RAG — tool functions for the query-time agent team.

Implements the 5-step retrieval/synthesis contract:
  1. vector_search       — broad recall from Qdrant (k=200)
  2. dedup_by_pointer    — deduplicate by unique (doc_id, node_id)
  3. rerank_by_hierarchical_path — LLM-based top-k reranker
  4. load_section        — read full .md slice via skeleton.json + char offsets
  5. synthesize          — LLM call producing a grounded answer with citations

See docs/requirements.md → Tool signatures (RAG).
See docs/requirements.md → Functional → Query-time agent team → 5-step contract.

NOTE: This module is intentionally free of akgentic actor imports so it can be
unit-tested without a running ActorSystem.  The ``vector_search`` function
accepts an injected ``search_fn`` callable so tests can mock the Qdrant call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from akgentic.tool.vector_store.protocol import SearchHit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

_SEP = "|"
"""Separator used to encode (doc_id, node_id) into SearchHit.ref_id."""


class RagSearchHit(BaseModel):
    """Enriched search hit carrying the decoded RAG payload fields.

    Wraps ``SearchHit`` with the decoded ``doc_id``, ``node_id``, and
    ``hierarchical_path`` extracted from the Qdrant payload stored in
    ``SearchHit.ref_id`` / ``SearchHit.text``.
    """

    doc_id: str
    node_id: str
    hierarchical_path: list[str] = Field(default_factory=list)
    source_path: str = ""
    char_start: int = 0
    char_end: int = 0
    text_preview: str = ""
    score: float = 0.0

    @classmethod
    def from_search_hit(cls, hit: SearchHit) -> "RagSearchHit":
        """Decode a ``SearchHit`` produced by the RAG indexing pipeline.

        The indexing pipeline stores ``doc_id|node_id`` in ``ref_id`` and
        a JSON-encoded ``QdrantPayload`` in ``text`` (or falls back to
        splitting ``ref_id``).
        """
        # ref_id encodes "doc_id|node_id"
        parts = hit.ref_id.split(_SEP, 1)
        doc_id = parts[0] if len(parts) >= 1 else hit.ref_id
        node_id = parts[1] if len(parts) == 2 else ""

        # text may be a JSON-encoded QdrantPayload or plain preview text
        hierarchical_path: list[str] = []
        source_path = ""
        char_start = 0
        char_end = 0
        text_preview = hit.text
        try:
            payload: dict[str, Any] = json.loads(hit.text)
            hierarchical_path = payload.get("hierarchical_path", [])
            source_path = payload.get("source_path", "")
            char_start = int(payload.get("char_start", 0))
            char_end = int(payload.get("char_end", 0))
            text_preview = payload.get("text_preview", hit.text)
        except (json.JSONDecodeError, TypeError):
            pass

        return cls(
            doc_id=doc_id,
            node_id=node_id,
            hierarchical_path=hierarchical_path,
            source_path=source_path,
            char_start=char_start,
            char_end=char_end,
            text_preview=text_preview,
            score=hit.score,
        )


class Section(BaseModel):
    """A loaded document section with its full text and citation key."""

    doc_id: str
    node_id: str
    hierarchical_path: list[str]
    text: str

    @property
    def citation_key(self) -> str:
        """Return the citation key used in synthesized answers."""
        return f"[{self.doc_id}#{self.node_id}]"


class Answer(BaseModel):
    """A grounded answer produced by the synthesizer."""

    text: str = Field(description="Answer text with inline [doc_id#node_id] citations")
    citations: list[str] = Field(
        default_factory=list,
        description="List of citation keys referenced in the answer",
    )


# ---------------------------------------------------------------------------
# Step 1 — vector_search
# ---------------------------------------------------------------------------


def vector_search(
    query: str,
    *,
    k: int = 200,
    collection: str = "proxy_pointer_rag",
    search_fn: Callable[[str, int, str], list[SearchHit]] | None = None,
) -> list[RagSearchHit]:
    """Search the Qdrant collection and return up to *k* hits.

    Args:
        query: Natural-language query string.
        k: Maximum number of hits to retrieve (default 200 for broad recall).
        collection: Qdrant collection name.
        search_fn: Optional injectable search callable for testing.  When
            ``None`` the function raises ``NotImplementedError`` — callers
            must inject a real backend or use the agent-level wiring.

    Returns:
        List of :class:`RagSearchHit` ordered by descending similarity score.
    """
    if search_fn is None:
        raise NotImplementedError(
            "vector_search requires an injected search_fn. "
            "Wire it via the VectorStoreActor proxy in the agent team."
        )
    raw_hits = search_fn(query, k, collection)
    return [RagSearchHit.from_search_hit(h) for h in raw_hits]


# ---------------------------------------------------------------------------
# Step 2 — dedup_by_pointer
# ---------------------------------------------------------------------------


def dedup_by_pointer(hits: list[RagSearchHit]) -> list[RagSearchHit]:
    """Deduplicate hits by unique ``(doc_id, node_id)`` pointer.

    When multiple chunks from the same section are returned by the vector
    search, only the highest-scoring chunk is kept.  Input order (by score
    descending) is preserved.

    Args:
        hits: List of search hits, typically from :func:`vector_search`.

    Returns:
        Deduplicated list with at most one entry per ``(doc_id, node_id)``.
    """
    seen: set[tuple[str, str]] = set()
    result: list[RagSearchHit] = []
    for hit in hits:
        key = (hit.doc_id, hit.node_id)
        if key not in seen:
            seen.add(key)
            result.append(hit)
    return result


# ---------------------------------------------------------------------------
# Step 3 — rerank_by_hierarchical_path
# ---------------------------------------------------------------------------

_RERANK_SYSTEM_PROMPT = """\
You are a relevance reranker for a legal/regulatory RAG system.
Given a user query and a list of document sections (identified by their
hierarchical path), select the {top_k} most relevant sections.

Return ONLY a JSON array of the selected section indices (0-based), ordered
from most to least relevant.  Example: [2, 0, 4, 1, 3]

Do not include any explanation or extra text — only the JSON array.
"""

_RERANK_USER_TEMPLATE = """\
Query: {query}

Sections:
{sections_text}

Select the {top_k} most relevant section indices as a JSON array.
"""


def rerank_by_hierarchical_path(
    query: str,
    hits: list[RagSearchHit],
    *,
    top_k: int = 5,
    llm_fn: Callable[[str, str], str] | None = None,
) -> list[RagSearchHit]:
    """Rerank *hits* by hierarchical path relevance using an LLM.

    The LLM receives the query and the hierarchical paths of all hits and
    returns the indices of the top-*k* most relevant sections.  Falls back
    to returning the first *top_k* hits by score if the LLM call fails.

    Args:
        query: The original user query.
        hits: Deduplicated hits from :func:`dedup_by_pointer`.
        top_k: Number of hits to return after reranking (default 5).
        llm_fn: Optional injectable ``(system_prompt, user_prompt) -> str``
            callable for testing.  When ``None`` the function raises
            ``NotImplementedError``.

    Returns:
        Up to *top_k* :class:`RagSearchHit` objects ordered by LLM relevance.
    """
    if not hits:
        return []

    effective_top_k = min(top_k, len(hits))

    if llm_fn is None:
        raise NotImplementedError(
            "rerank_by_hierarchical_path requires an injected llm_fn. "
            "Wire it via the LLM model client in the agent team."
        )

    sections_text = "\n".join(
        f"[{i}] {' > '.join(hit.hierarchical_path) or hit.node_id} "
        f"(score={hit.score:.3f})"
        for i, hit in enumerate(hits)
    )
    system_prompt = _RERANK_SYSTEM_PROMPT.format(top_k=effective_top_k)
    user_prompt = _RERANK_USER_TEMPLATE.format(
        query=query,
        sections_text=sections_text,
        top_k=effective_top_k,
    )

    try:
        raw = llm_fn(system_prompt, user_prompt)
        indices: list[int] = json.loads(raw.strip())
        # Validate and clamp indices
        valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(hits)]
        # Deduplicate while preserving order
        seen_idx: set[int] = set()
        ordered: list[int] = []
        for i in valid:
            if i not in seen_idx:
                seen_idx.add(i)
                ordered.append(i)
        return [hits[i] for i in ordered[:effective_top_k]]
    except Exception:
        logger.exception(
            "rerank_by_hierarchical_path: LLM rerank failed; falling back to top-%d by score",
            effective_top_k,
        )
        return hits[:effective_top_k]


# ---------------------------------------------------------------------------
# Step 4 — load_section
# ---------------------------------------------------------------------------


def load_section(
    doc_id: str,
    node_id: str,
    *,
    md_root: Path,
    skeleton_root: Path | None = None,
    hits: list[RagSearchHit] | None = None,
) -> str:
    """Load the full text of a document section from disk.

    Reads the source ``.md`` file slice using the char offsets stored in the
    skeleton tree (``skeleton.json``) or, as a convenience shortcut, from the
    ``hits`` list when the caller already has the offsets in memory.

    Args:
        doc_id: Document identifier (stable hash of relative path).
        node_id: Section node identifier.
        md_root: Root directory containing the parsed ``.md`` files.
        skeleton_root: Root directory containing ``skeleton.json`` files.
            Defaults to *md_root* when ``None``.
        hits: Optional list of :class:`RagSearchHit` to look up offsets from
            without re-reading ``skeleton.json``.

    Returns:
        The full text of the section as a string.

    Raises:
        FileNotFoundError: If the ``.md`` or ``skeleton.json`` file cannot be
            found.
        KeyError: If *node_id* is not present in the skeleton tree.
    """
    if skeleton_root is None:
        skeleton_root = md_root

    # Fast path: offsets already available in hits list
    if hits is not None:
        for hit in hits:
            if hit.doc_id == doc_id and hit.node_id == node_id and hit.source_path:
                md_path = md_root / hit.source_path
                text = md_path.read_text(encoding="utf-8")
                return text[hit.char_start : hit.char_end]

    # Slow path: look up offsets from skeleton.json
    skeleton_path = skeleton_root / doc_id / "skeleton.json"
    if not skeleton_path.exists():
        # Try flat layout: skeleton.json lives next to the .md file
        # Search for any skeleton.json that matches doc_id
        candidates = list(skeleton_root.rglob("skeleton.json"))
        skeleton_path_found: Path | None = None
        for c in candidates:
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
                if data.get("doc_id") == doc_id:
                    skeleton_path_found = c
                    break
            except Exception:
                continue
        if skeleton_path_found is None:
            raise FileNotFoundError(
                f"load_section: skeleton.json not found for doc_id={doc_id!r}"
            )
        skeleton_path = skeleton_path_found

    skeleton_data = json.loads(skeleton_path.read_text(encoding="utf-8"))
    source_path: str | None = skeleton_data.get("source_path")
    if source_path is None:
        raise KeyError(f"load_section: 'source_path' missing in skeleton.json for doc_id={doc_id!r}")

    # Find the node in the flattened tree
    char_start, char_end = _find_node_offsets(skeleton_data.get("nodes", []), node_id)

    md_path = md_root / source_path
    text = md_path.read_text(encoding="utf-8")
    return text[char_start:char_end]


def _find_node_offsets(nodes: list[dict[str, Any]], node_id: str) -> tuple[int, int]:
    """Recursively search *nodes* for *node_id* and return its char offsets."""
    for node in nodes:
        if node.get("node_id") == node_id:
            return int(node["char_start"]), int(node["char_end"])
        children = node.get("children", [])
        if children:
            result = _find_node_offsets(children, node_id)
            if result != (-1, -1):
                return result
    return -1, -1


# ---------------------------------------------------------------------------
# Step 5 — synthesize
# ---------------------------------------------------------------------------

_SYNTHESIZE_SYSTEM_PROMPT = """\
You are a precise legal/regulatory research assistant.
Answer the user's question using ONLY the provided document sections.
Every factual claim MUST be supported by an inline citation in the format
[doc_id#node_id] immediately after the relevant sentence.

Rules:
- Do not fabricate information not present in the sections.
- If the sections do not contain enough information, say so explicitly.
- Keep the answer concise and well-structured.
- List all citation keys used at the end under "## References".
"""

_SYNTHESIZE_USER_TEMPLATE = """\
Question: {query}

Document sections:
{sections_text}

Provide a grounded answer with [doc_id#node_id] citations.
"""


def synthesize(
    query: str,
    sections: list[Section],
    *,
    llm_fn: Callable[[str, str], str] | None = None,
) -> Answer:
    """Produce a grounded answer from the loaded sections using an LLM.

    Args:
        query: The original user query.
        sections: Loaded document sections from :func:`load_section`.
        llm_fn: Optional injectable ``(system_prompt, user_prompt) -> str``
            callable for testing.  When ``None`` the function raises
            ``NotImplementedError``.

    Returns:
        An :class:`Answer` with inline citations and a deduplicated citation list.
    """
    if llm_fn is None:
        raise NotImplementedError(
            "synthesize requires an injected llm_fn. "
            "Wire it via the LLM model client in the agent team."
        )

    sections_text = "\n\n".join(
        f"--- {s.citation_key} ({' > '.join(s.hierarchical_path)}) ---\n{s.text}"
        for s in sections
    )
    user_prompt = _SYNTHESIZE_USER_TEMPLATE.format(
        query=query,
        sections_text=sections_text,
    )

    answer_text = llm_fn(_SYNTHESIZE_SYSTEM_PROMPT, user_prompt)

    # Extract citation keys from the answer text
    import re
    citations = list(dict.fromkeys(re.findall(r"\[[^\]]+#[^\]]+\]", answer_text)))

    return Answer(text=answer_text, citations=citations)
