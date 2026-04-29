"""Unit tests for proxy_pointer_rag.indexing.parse."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proxy_pointer_rag.indexing.parse import doc_id_for, run_parse


class TestDocIdFor:
    def test_stable(self) -> None:
        assert doc_id_for("subdir/file.pdf") == doc_id_for("subdir/file.pdf")

    def test_is_sha1_hex(self) -> None:
        result = doc_id_for("a/b.pdf")
        expected = hashlib.sha1("a/b.pdf".encode()).hexdigest()
        assert result == expected

    def test_differs_by_path(self) -> None:
        assert doc_id_for("a.pdf") != doc_id_for("b.pdf")


class TestRunParse:
    def _make_converter(self, md_text: str = "# Hello\n\nWorld.") -> MagicMock:
        doc = MagicMock()
        doc.export_to_markdown.return_value = md_text
        result = MagicMock()
        result.document = doc
        converter = MagicMock()
        converter.convert.return_value = result
        return converter

    def test_converts_supported_file(self, tmp_path: Path) -> None:
        src = tmp_path / "input"
        src.mkdir()
        (src / "report.pdf").write_bytes(b"%PDF fake")
        out = tmp_path / "output"

        converter = self._make_converter("# Title\n\nContent.")

        with patch("docling.document_converter.DocumentConverter", return_value=converter):
            run_parse(str(src), str(out))

        md_file = out / "report.md"
        assert md_file.exists()
        assert md_file.read_text() == "# Title\n\nContent."

    def test_skips_already_converted_by_default(self, tmp_path: Path) -> None:
        src = tmp_path / "input"
        src.mkdir()
        (src / "doc.pdf").write_bytes(b"%PDF fake")
        out = tmp_path / "output"
        out.mkdir()
        existing = out / "doc.md"
        existing.write_text("existing content")

        converter = self._make_converter("new content")

        with patch("docling.document_converter.DocumentConverter", return_value=converter):
            run_parse(str(src), str(out), force=False)

        assert existing.read_text() == "existing content"
        converter.convert.assert_not_called()

    def test_force_reconverts(self, tmp_path: Path) -> None:
        src = tmp_path / "input"
        src.mkdir()
        (src / "doc.pdf").write_bytes(b"%PDF fake")
        out = tmp_path / "output"
        out.mkdir()
        existing = out / "doc.md"
        existing.write_text("old content")

        converter = self._make_converter("new content")

        with patch("docling.document_converter.DocumentConverter", return_value=converter):
            run_parse(str(src), str(out), force=True)

        assert existing.read_text() == "new content"

    def test_preserves_relative_tree(self, tmp_path: Path) -> None:
        src = tmp_path / "input"
        (src / "sub").mkdir(parents=True)
        (src / "sub" / "nested.docx").write_bytes(b"fake docx")
        out = tmp_path / "output"

        converter = self._make_converter("nested md")

        with patch("docling.document_converter.DocumentConverter", return_value=converter):
            run_parse(str(src), str(out))

        assert (out / "sub" / "nested.md").exists()

    def test_ignores_unsupported_extensions(self, tmp_path: Path) -> None:
        src = tmp_path / "input"
        src.mkdir()
        (src / "image.png").write_bytes(b"fake png")
        out = tmp_path / "output"

        converter = self._make_converter()

        with patch("docling.document_converter.DocumentConverter", return_value=converter):
            run_parse(str(src), str(out))

        converter.convert.assert_not_called()

    def test_raises_on_missing_input_dir(self, tmp_path: Path) -> None:
        with patch("docling.document_converter.DocumentConverter"):
            with pytest.raises(ValueError, match="input-dir"):
                run_parse(str(tmp_path / "nonexistent"), str(tmp_path / "out"))

    def test_raises_on_missing_docling(self, tmp_path: Path) -> None:
        src = tmp_path / "input"
        src.mkdir()
        with patch.dict("sys.modules", {"docling": None, "docling.document_converter": None}):
            with pytest.raises(ImportError, match="docling"):
                run_parse(str(src), str(tmp_path / "out"))
