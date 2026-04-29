"""Unit tests for proxy_pointer_rag.indexing.__main__ (argparse dispatcher)."""

import pytest
from unittest.mock import MagicMock, patch

from proxy_pointer_rag.indexing.__main__ import _build_parser, main


class TestBuildParser:
    def test_parse_subcommand_required_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["parse", "--input-dir", "/in", "--output-dir", "/out"])
        assert args.subcommand == "parse"
        assert args.input_dir == "/in"
        assert args.output_dir == "/out"
        assert args.force is False

    def test_parse_force_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["parse", "--input-dir", "/in", "--output-dir", "/out", "--force"])
        assert args.force is True

    def test_skeleton_subcommand(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["skeleton", "--input-dir", "/md", "--output-dir", "/sk"])
        assert args.subcommand == "skeleton"
        assert args.input_dir == "/md"
        assert args.output_dir == "/sk"

    def test_index_subcommand_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "index",
            "--md-dir", "/md",
            "--skeleton-dir", "/sk",
            "--collection", "my-col",
        ])
        assert args.subcommand == "index"
        assert args.qdrant_url == "http://localhost:6333"
        assert args.qdrant_api_key is None
        assert args.model == "gpt-4.1-mini"

    def test_index_subcommand_custom_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "index",
            "--md-dir", "/md",
            "--skeleton-dir", "/sk",
            "--collection", "col",
            "--qdrant-url", "http://remote:6333",
            "--qdrant-api-key", "secret",
            "--model", "gpt-4o",
        ])
        assert args.qdrant_url == "http://remote:6333"
        assert args.qdrant_api_key == "secret"
        assert args.model == "gpt-4o"

    def test_missing_subcommand_exits(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestMain:
    def test_dispatches_parse(self) -> None:
        mock_run = MagicMock()
        with patch("proxy_pointer_rag.indexing.parse.run_parse", mock_run):
            main(["parse", "--input-dir", "/in", "--output-dir", "/out"])
        mock_run.assert_called_once_with(input_dir="/in", output_dir="/out", force=False)

    def test_dispatches_parse_with_force(self) -> None:
        mock_run = MagicMock()
        with patch("proxy_pointer_rag.indexing.parse.run_parse", mock_run):
            main(["parse", "--input-dir", "/in", "--output-dir", "/out", "--force"])
        mock_run.assert_called_once_with(input_dir="/in", output_dir="/out", force=True)
