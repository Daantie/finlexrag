"""Proxy-Pointer RAG — skeleton generation stage.

Parses Markdown heading structure from converted .md files and writes one
``skeleton.json`` per document.  The skeleton is a tree of ``SkeletonNode``
objects serialised as JSON.  The stage is deterministic and pure-text: no LLM
calls, no network I/O.

See docs/requirements.md → Functional requirements → Indexing → ``skeleton``.
"""

import json
import logging
import re
from pathlib import Path

from proxy_pointer_rag.indexing.models import SkeletonNode
from proxy_pointer_rag.indexing.parse import doc_id_for

logger = logging.getLogger(__name__)

# Matches ATX headings: optional leading whitespace, 1-6 '#', space, title
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _parse_headings(text: str) -> list[tuple[int, str, int]]:
    """Return a list of (level, title, char_offset) for every ATX heading.

    Args:
        text: Full Markdown document text.

    Returns:
        Ordered list of (heading_level, heading_title, char_offset_of_line).
    """
    headings: list[tuple[int, str, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        m = _HEADING_RE.match(line.rstrip("\n\r"))
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((level, title, offset))
        offset += len(line)
    return headings


def _build_tree(
    doc_id: str,
    text: str,
    headings: list[tuple[int, str, int]],
) -> list[SkeletonNode]:
    """Build a forest of ``SkeletonNode`` from a flat heading list.

    Each node's ``char_end`` is set to the start of the next sibling/parent
    heading (or end-of-document).

    Args:
        doc_id: Stable document identifier.
        text: Full Markdown text (used to determine ``char_end``).
        headings: Ordered list of (level, title, char_start).

    Returns:
        Top-level ``SkeletonNode`` list (the forest roots).
    """
    doc_len = len(text)

    # Stack entries: (node, path_so_far)
    stack: list[tuple[SkeletonNode, list[str]]] = []
    roots: list[SkeletonNode] = []

    for i, (level, title, char_start) in enumerate(headings):
        # Determine char_end: start of next heading or end of document
        char_end = headings[i + 1][2] if i + 1 < len(headings) else doc_len

        # Build hierarchical path
        # Pop stack entries that are at the same level or deeper
        while stack and stack[-1][0].level >= level:
            stack.pop()

        parent_path = stack[-1][1] if stack else []
        path = parent_path + [title]

        node = SkeletonNode(
            node_id=SkeletonNode.make_node_id(doc_id, path),
            title=title,
            level=level,
            path=path,
            char_start=char_start,
            char_end=char_end,
        )

        if stack:
            stack[-1][0].children.append(node)
        else:
            roots.append(node)

        stack.append((node, path))

    return roots


def _skeleton_for_document(md_path: Path, rel: Path) -> list[SkeletonNode]:
    """Parse a single .md file and return its skeleton forest.

    Args:
        md_path: Absolute path to the .md file.
        rel: Relative path used to derive ``doc_id``.

    Returns:
        List of top-level ``SkeletonNode`` objects.
    """
    text = md_path.read_text(encoding="utf-8")
    did = doc_id_for(str(rel))
    headings = _parse_headings(text)
    return _build_tree(did, text, headings)


def run_skeleton(input_dir: str, output_dir: str) -> None:
    """Parse all .md files under *input_dir* and write ``skeleton.json`` files.

    The relative directory tree of *input_dir* is preserved under *output_dir*.
    Each ``skeleton.json`` contains a JSON array of serialised ``SkeletonNode``
    objects (the top-level forest).  The stage is idempotent: re-running it
    overwrites existing ``skeleton.json`` files with identical content.

    Args:
        input_dir: Directory containing .md files (output of the ``parse`` stage).
        output_dir: Directory where ``skeleton.json`` files will be written.
    """
    src_root = Path(input_dir).resolve()
    out_root = Path(output_dir).resolve()

    if not src_root.is_dir():
        raise ValueError(f"input-dir does not exist or is not a directory: {src_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    md_files = sorted(p for p in src_root.rglob("*.md") if p.is_file())

    if not md_files:
        logger.warning("No .md files found under %s", src_root)
        return

    logger.info("Found %d .md file(s) under %s", len(md_files), src_root)

    processed = errors = 0

    for md_path in md_files:
        rel = md_path.relative_to(src_root)
        out_path = out_root / rel.parent / (rel.stem + ".skeleton.json")

        try:
            roots = _skeleton_for_document(md_path, rel)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [node.model_dump() for node in roots]
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(
                "skeleton: %s → %s  (%d top-level node(s))",
                rel,
                out_path.relative_to(out_root),
                len(roots),
            )
            processed += 1
        except Exception:
            logger.exception("Failed to build skeleton for %s", md_path)
            errors += 1

    logger.info("skeleton complete — processed=%d  errors=%d", processed, errors)
