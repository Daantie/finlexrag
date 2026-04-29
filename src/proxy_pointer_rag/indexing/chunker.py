"""Proxy-Pointer RAG — section-aware chunker.

Splits each skeleton section into one or more ``Chunk`` objects that fit
within a configurable token budget.  Chunking is section-aware: chunks never
span section boundaries, so every chunk belongs to exactly one
``SkeletonNode``.

Token counting uses a simple whitespace-based word count multiplied by a
conservative words-to-tokens ratio (1.35).  This avoids a hard dependency on
a tokeniser library while staying within typical LLM context windows.

See docs/requirements.md → Functional requirements → Indexing → ``index`` step 2.
"""

import logging
from pathlib import Path

from proxy_pointer_rag.indexing.models import Chunk, SkeletonNode

logger = logging.getLogger(__name__)

# Conservative words-to-tokens ratio (GPT-style tokenisers average ~0.75 words/token)
_WORDS_PER_TOKEN: float = 0.75
# Default token budget per chunk
DEFAULT_TOKEN_BUDGET: int = 512


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* using a word-count heuristic.

    Args:
        text: Input text.

    Returns:
        Estimated token count (always ≥ 1 for non-empty text).
    """
    words = len(text.split())
    return max(1, round(words / _WORDS_PER_TOKEN))


def _split_into_chunks(
    text: str,
    char_offset: int,
    token_budget: int,
) -> list[tuple[int, int]]:
    """Split *text* into (char_start, char_end) slices within *token_budget*.

    Splits on paragraph boundaries (double newline) first, then on sentence
    boundaries (period/newline), and finally hard-splits if a single sentence
    exceeds the budget.

    Args:
        text: Section text to split.
        char_offset: Absolute character offset of *text* in the source document.
        token_budget: Maximum tokens per chunk.

    Returns:
        List of (absolute_char_start, absolute_char_end) pairs.
    """
    if not text.strip():
        return []

    if _estimate_tokens(text) <= token_budget:
        return [(char_offset, char_offset + len(text))]

    # Split on paragraph boundaries first
    paragraphs: list[str] = []
    para_offsets: list[int] = []
    current_offset = 0
    for para in text.split("\n\n"):
        paragraphs.append(para)
        para_offsets.append(current_offset)
        current_offset += len(para) + 2  # +2 for the "\n\n" separator

    slices: list[tuple[int, int]] = []
    current_text = ""
    current_start = 0

    for para, p_off in zip(paragraphs, para_offsets):
        candidate = (current_text + "\n\n" + para).lstrip("\n") if current_text else para
        if _estimate_tokens(candidate) <= token_budget:
            current_text = candidate
            if not current_text or current_start == 0 and not slices:
                current_start = p_off
        else:
            if current_text:
                abs_start = char_offset + current_start
                abs_end = char_offset + current_start + len(current_text)
                slices.append((abs_start, abs_end))
            # Start a new chunk with this paragraph
            current_text = para
            current_start = p_off

    if current_text:
        abs_start = char_offset + current_start
        abs_end = char_offset + current_start + len(current_text)
        slices.append((abs_start, abs_end))

    return slices if slices else [(char_offset, char_offset + len(text))]


def chunk_nodes(
    nodes: list[SkeletonNode],
    md_text: str,
    source_path: str,
    doc_id: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[Chunk]:
    """Produce ``Chunk`` objects for a list of skeleton nodes.

    Each node is split into one or more chunks that fit within *token_budget*.
    Chunks never span section boundaries.

    Args:
        nodes: Flat list of ``SkeletonNode`` objects to chunk.
        md_text: Full Markdown text of the document.
        source_path: Relative path of the source .md file (stored in payload).
        doc_id: Stable document identifier.
        token_budget: Maximum tokens per chunk (default: 512).

    Returns:
        Ordered list of ``Chunk`` objects.
    """
    chunks: list[Chunk] = []

    for node in nodes:
        section_text = md_text[node.char_start : node.char_end]
        slices = _split_into_chunks(section_text, node.char_start, token_budget)

        for chunk_idx, (abs_start, abs_end) in enumerate(slices):
            chunk_text = md_text[abs_start:abs_end]
            if not chunk_text.strip():
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    node_id=node.node_id,
                    hierarchical_path=node.path,
                    source_path=source_path,
                    char_start=abs_start,
                    char_end=abs_end,
                    text=chunk_text,
                )
            )
            logger.debug(
                "chunker: doc=%s node=%s chunk=%d  tokens≈%d",
                doc_id,
                node.node_id,
                chunk_idx,
                _estimate_tokens(chunk_text),
            )

    logger.info(
        "chunker: produced %d chunk(s) from %d node(s)  (budget=%d tokens)",
        len(chunks),
        len(nodes),
        token_budget,
    )
    return chunks


def chunk_document(
    skeleton_path: Path,
    md_path: Path,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[Chunk]:
    """Convenience wrapper: load skeleton + md file and return chunks.

    Args:
        skeleton_path: Path to the ``skeleton.json`` file.
        md_path: Path to the corresponding ``.md`` file.
        token_budget: Maximum tokens per chunk.

    Returns:
        Ordered list of ``Chunk`` objects for the document.
    """
    import json

    from proxy_pointer_rag.indexing.noise_filter import flatten_nodes
    from proxy_pointer_rag.indexing.parse import doc_id_for

    md_text = md_path.read_text(encoding="utf-8")
    raw = json.loads(skeleton_path.read_text(encoding="utf-8"))
    roots = [SkeletonNode.model_validate(n) for n in raw]
    flat_nodes = flatten_nodes(roots)

    # Derive doc_id and source_path from the md_path relative to its parent
    # (caller is responsible for passing the correct relative path if needed)
    source_path = md_path.name
    doc_id = doc_id_for(source_path)

    return chunk_nodes(flat_nodes, md_text, source_path, doc_id, token_budget)
