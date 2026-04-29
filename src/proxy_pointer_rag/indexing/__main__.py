"""Proxy-Pointer RAG — offline indexing CLI.

Usage:
    python -m proxy_pointer_rag.indexing parse    --input-dir <dir> --output-dir <dir> [--force]
    python -m proxy_pointer_rag.indexing skeleton --input-dir <dir> --output-dir <dir>
    python -m proxy_pointer_rag.indexing index    --md-dir <dir> --skeleton-dir <dir> --collection <name>

See docs/requirements.md → Functional requirements → Indexing pipeline.
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m proxy_pointer_rag.indexing",
        description="Proxy-Pointer RAG offline indexing pipeline.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ── parse ──────────────────────────────────────────────────────────────
    parse_p = sub.add_parser(
        "parse",
        help="Convert source documents (.docx/.pdf/.pptx/…) to Markdown via docling.",
    )
    parse_p.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Directory containing source documents.",
    )
    parse_p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where converted .md files will be written (tree is preserved).",
    )
    parse_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-convert files that have already been converted (default: skip-if-exists).",
    )

    # ── skeleton ───────────────────────────────────────────────────────────
    skeleton_p = sub.add_parser(
        "skeleton",
        help="Parse .md files and write one skeleton.json per document.",
    )
    skeleton_p.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Directory containing .md files (output of `parse`).",
    )
    skeleton_p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where skeleton.json files will be written.",
    )

    # ── index ──────────────────────────────────────────────────────────────
    index_p = sub.add_parser(
        "index",
        help="Noise-filter, chunk, embed, and upsert sections into Qdrant.",
    )
    index_p.add_argument(
        "--md-dir",
        required=True,
        metavar="DIR",
        help="Directory containing .md files.",
    )
    index_p.add_argument(
        "--skeleton-dir",
        required=True,
        metavar="DIR",
        help="Directory containing skeleton.json files (output of `skeleton`).",
    )
    index_p.add_argument(
        "--collection",
        required=True,
        metavar="NAME",
        help="Qdrant collection name to upsert into.",
    )
    index_p.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        metavar="URL",
        help="Qdrant server URL (default: http://localhost:6333).",
    )
    index_p.add_argument(
        "--qdrant-api-key",
        default=None,
        metavar="KEY",
        help="Qdrant API key (optional).",
    )
    index_p.add_argument(
        "--model",
        default="gpt-4.1-mini",
        metavar="MODEL",
        help="LLM model used for noise filtering (default: gpt-4.1-mini).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "parse":
        from proxy_pointer_rag.indexing.parse import run_parse

        run_parse(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            force=args.force,
        )

    elif args.subcommand == "skeleton":
        from proxy_pointer_rag.indexing.skeleton import run_skeleton

        run_skeleton(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

    elif args.subcommand == "index":
        from proxy_pointer_rag.indexing.embed_and_upsert import run_index

        run_index(
            md_dir=args.md_dir,
            skeleton_dir=args.skeleton_dir,
            collection=args.collection,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            model=args.model,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
