"""Proxy-Pointer RAG — LLM-based noise filter for skeleton sections.

Drops sections that are unlikely to contain substantive legal/regulatory
content: tables of contents, abbreviation lists, reference lists, blank
sections, boilerplate headers, etc.

The filter uses a cheap LLM (default: ``gpt-4.1-mini``) to keep costs low.
Each section is evaluated independently; the LLM returns a simple boolean
decision.

See docs/requirements.md → Functional requirements → Indexing → ``index`` step 1.
See docs/requirements.md → Risks → LLM cost of filter + reranker.

NOTE: This module is async; call ``filter_nodes`` with ``asyncio.run`` or
from an existing event loop.
"""

import asyncio
import logging
from typing import Any

from proxy_pointer_rag.indexing.models import SkeletonNode

logger = logging.getLogger(__name__)

# System prompt for the noise-filter LLM
_SYSTEM_PROMPT = """\
You are a document quality filter for a legal/regulatory RAG system.
Your task: decide whether a document section contains substantive content
worth indexing, or whether it is noise that should be dropped.

Drop the section (respond with "drop") if it is primarily:
- A table of contents or navigation list
- An abbreviation or acronym list
- A bibliography, references, or footnotes list
- A blank or near-blank section (fewer than 20 meaningful words)
- A boilerplate cover page, disclaimer, or signature block
- A list of figures, tables, or annexes

Keep the section (respond with "keep") if it contains:
- Substantive regulatory, legal, or policy text
- Definitions with explanatory prose
- Procedural or operational requirements
- Any section with meaningful analytical content

Respond with exactly one word: "keep" or "drop". No explanation.
"""


async def _classify_section(
    agent: Any,
    node: SkeletonNode,
    text_snippet: str,
) -> bool:
    """Return True if the section should be kept, False if it should be dropped.

    Args:
        agent: A pydantic-ai ``Agent`` instance configured for the filter model.
        node: The skeleton node being evaluated.
        text_snippet: The first ~512 chars of the section text.

    Returns:
        True to keep, False to drop.
    """
    prompt = (
        f"Section path: {' > '.join(node.path)}\n"
        f"Section title: {node.title}\n\n"
        f"Section text (first 512 chars):\n{text_snippet[:512]}\n\n"
        "Keep or drop?"
    )
    try:
        result = await agent.run(prompt)
        decision = str(result.output).strip().lower()
        return not decision.startswith("drop")
    except Exception:
        logger.exception(
            "noise_filter: LLM call failed for node '%s'; defaulting to keep",
            node.node_id,
        )
        return True


def flatten_nodes(nodes: list[SkeletonNode]) -> list[SkeletonNode]:
    """Return a flat list of all nodes in the tree (depth-first).

    Args:
        nodes: Top-level skeleton nodes.

    Returns:
        All nodes in depth-first order.
    """
    result: list[SkeletonNode] = []
    for node in nodes:
        result.append(node)
        result.extend(flatten_nodes(node.children))
    return result


async def filter_nodes(
    nodes: list[SkeletonNode],
    md_text: str,
    model: str = "gpt-4.1-mini",
) -> list[SkeletonNode]:
    """Filter a flat list of skeleton nodes, dropping noise sections.

    Only leaf-level and meaningful intermediate nodes are evaluated.
    The function preserves the original order of kept nodes.

    Args:
        nodes: Flat list of ``SkeletonNode`` objects to evaluate.
        md_text: Full Markdown text of the document (used to extract snippets).
        model: LLM model identifier (default: ``gpt-4.1-mini`` for cost control).

    Returns:
        Filtered list containing only nodes classified as substantive.
    """
    try:
        from pydantic_ai import Agent

        from akgentic.llm import ModelConfig
        from akgentic.llm.providers import create_model
    except ImportError as exc:
        raise ImportError(
            "pydantic-ai and akgentic-llm are required for noise filtering."
        ) from exc

    model_cfg = ModelConfig(provider="openai", model=model)
    pydantic_model = create_model(model_cfg)
    agent: Any = Agent(pydantic_model, system_prompt=_SYSTEM_PROMPT)

    kept: list[SkeletonNode] = []
    dropped = 0

    for node in nodes:
        snippet = md_text[node.char_start : node.char_end]
        should_keep = await _classify_section(agent, node, snippet)
        if should_keep:
            kept.append(node)
        else:
            logger.debug(
                "noise_filter: dropped node '%s' (path=%s)",
                node.node_id,
                " > ".join(node.path),
            )
            dropped += 1

    logger.info(
        "noise_filter: kept=%d  dropped=%d  (model=%s)",
        len(kept),
        dropped,
        model,
    )
    return kept


def filter_nodes_sync(
    nodes: list[SkeletonNode],
    md_text: str,
    model: str = "gpt-4.1-mini",
) -> list[SkeletonNode]:
    """Synchronous wrapper around :func:`filter_nodes`.

    Args:
        nodes: Flat list of ``SkeletonNode`` objects to evaluate.
        md_text: Full Markdown text of the document.
        model: LLM model identifier.

    Returns:
        Filtered list containing only nodes classified as substantive.
    """
    return asyncio.run(filter_nodes(nodes, md_text, model))
