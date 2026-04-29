"""Proxy-Pointer RAG — document parsing stage.

Wraps docling's DocumentConverter to convert source documents (.docx, .pdf,
.pptx, …) to Markdown, preserving the relative directory tree.

Each output .md file is named after the source file with a `.md` extension.
The `doc_id` for a document is the SHA-1 of its relative path (UTF-8), giving
a stable, path-derived identifier across reruns.

See docs/requirements.md → Functional requirements → Indexing → `parse`.

NOTE: docling is an acknowledged AGENTS.md exception — it is declared as the
optional extra [docs_docling] in packages/akgentic-tool/pyproject.toml.
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions that docling can handle
_SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt"}
)


def doc_id_for(relative_path: str) -> str:
    """Return a stable SHA-1 hex digest for *relative_path* (UTF-8 encoded)."""
    return hashlib.sha1(relative_path.encode()).hexdigest()


def run_parse(input_dir: str, output_dir: str, force: bool = False) -> None:
    """Convert all supported documents under *input_dir* to Markdown.

    Args:
        input_dir: Root directory containing source documents.
        output_dir: Root directory where .md files will be written.
            The relative sub-tree of *input_dir* is preserved.
        force: When True, re-convert files whose .md output already exists.
            When False (default), skip already-converted files (idempotent).
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "docling is required for the parse stage. "
            "Install it with: uv sync --extra docs_docling"
        ) from exc

    src_root = Path(input_dir).resolve()
    out_root = Path(output_dir).resolve()

    if not src_root.is_dir():
        raise ValueError(f"input-dir does not exist or is not a directory: {src_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()

    source_files = [
        p for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    ]

    if not source_files:
        logger.warning("No supported source files found under %s", src_root)
        return

    logger.info("Found %d source file(s) under %s", len(source_files), src_root)

    converted = skipped = errors = 0

    for src_path in sorted(source_files):
        rel = src_path.relative_to(src_root)
        out_path = out_root / rel.with_suffix(".md")
        did = doc_id_for(str(rel))

        if out_path.exists() and not force:
            logger.debug("Skipping (already converted): %s  [doc_id=%s]", rel, did)
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Converting: %s  [doc_id=%s]", rel, did)
            result = converter.convert(str(src_path))
            md_text: str = result.document.export_to_markdown()
            out_path.write_text(md_text, encoding="utf-8")
            converted += 1
        except Exception:
            logger.exception("Failed to convert %s", src_path)
            errors += 1

    logger.info(
        "parse complete — converted=%d  skipped=%d  errors=%d",
        converted,
        skipped,
        errors,
    )
