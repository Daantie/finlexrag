"""Unit tests for proxy_pointer_rag.indexing.skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proxy_pointer_rag.indexing.models import SkeletonNode
from proxy_pointer_rag.indexing.skeleton import (
    _build_tree,
    _parse_headings,
    _skeleton_for_document,
    run_skeleton,
)


# ---------------------------------------------------------------------------
# _parse_headings
# ---------------------------------------------------------------------------


class TestParseHeadings:
    def test_empty_document(self) -> None:
        assert _parse_headings("") == []

    def test_no_headings(self) -> None:
        assert _parse_headings("Just some text\nNo headings here.") == []

    def test_single_h1(self) -> None:
        text = "# Title\nsome content"
        headings = _parse_headings(text)
        assert len(headings) == 1
        level, title, offset = headings[0]
        assert level == 1
        assert title == "Title"
        assert offset == 0

    def test_multiple_levels(self) -> None:
        text = "# H1\n## H2\n### H3\n"
        headings = _parse_headings(text)
        assert [h[0] for h in headings] == [1, 2, 3]
        assert [h[1] for h in headings] == ["H1", "H2", "H3"]

    def test_char_offsets_are_correct(self) -> None:
        text = "# First\n## Second\n"
        headings = _parse_headings(text)
        assert headings[0][2] == 0
        assert headings[1][2] == len("# First\n")

    def test_ignores_non_atx_headings(self) -> None:
        text = "Title\n=====\nSubtitle\n--------\n"
        assert _parse_headings(text) == []

    def test_strips_trailing_whitespace_from_title(self) -> None:
        text = "# Title   \n"
        headings = _parse_headings(text)
        assert headings[0][1] == "Title"


# ---------------------------------------------------------------------------
# _build_tree
# ---------------------------------------------------------------------------


class TestBuildTree:
    def test_empty_headings_returns_empty(self) -> None:
        assert _build_tree("doc1", "text", []) == []

    def test_single_root_node(self) -> None:
        text = "# Root\ncontent"
        headings = [(1, "Root", 0)]
        roots = _build_tree("doc1", text, headings)
        assert len(roots) == 1
        assert roots[0].title == "Root"
        assert roots[0].level == 1
        assert roots[0].path == ["Root"]

    def test_nested_children(self) -> None:
        text = "# A\n## B\n### C\n"
        headings = _parse_headings(text)
        roots = _build_tree("doc1", text, headings)
        assert len(roots) == 1
        assert roots[0].title == "A"
        assert len(roots[0].children) == 1
        assert roots[0].children[0].title == "B"
        assert len(roots[0].children[0].children) == 1
        assert roots[0].children[0].children[0].title == "C"

    def test_sibling_nodes_at_same_level(self) -> None:
        text = "# A\n# B\n# C\n"
        headings = _parse_headings(text)
        roots = _build_tree("doc1", text, headings)
        assert len(roots) == 3
        assert [r.title for r in roots] == ["A", "B", "C"]

    def test_char_end_set_to_next_heading_start(self) -> None:
        text = "# A\ncontent A\n# B\ncontent B\n"
        headings = _parse_headings(text)
        roots = _build_tree("doc1", text, headings)
        assert roots[0].char_end == headings[1][2]

    def test_last_node_char_end_is_doc_length(self) -> None:
        text = "# A\ncontent"
        headings = _parse_headings(text)
        roots = _build_tree("doc1", text, headings)
        assert roots[0].char_end == len(text)

    def test_node_ids_are_stable(self) -> None:
        text = "# A\n## B\n"
        headings = _parse_headings(text)
        roots1 = _build_tree("doc1", text, headings)
        roots2 = _build_tree("doc1", text, headings)
        assert roots1[0].node_id == roots2[0].node_id
        assert roots1[0].children[0].node_id == roots2[0].children[0].node_id

    def test_hierarchical_path_includes_ancestors(self) -> None:
        text = "# A\n## B\n### C\n"
        headings = _parse_headings(text)
        roots = _build_tree("doc1", text, headings)
        leaf = roots[0].children[0].children[0]
        assert leaf.path == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# _skeleton_for_document
# ---------------------------------------------------------------------------


class TestSkeletonForDocument:
    def test_returns_skeleton_nodes(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nSome content.\n\n## Section\n\nMore content.\n")
        nodes = _skeleton_for_document(md, Path("doc.md"))
        assert len(nodes) == 1
        assert nodes[0].title == "Title"
        assert len(nodes[0].children) == 1

    def test_empty_document_returns_empty(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("")
        nodes = _skeleton_for_document(md, Path("empty.md"))
        assert nodes == []


# ---------------------------------------------------------------------------
# run_skeleton
# ---------------------------------------------------------------------------


class TestRunSkeleton:
    def test_writes_skeleton_json(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        (src / "doc.md").write_text("# Title\n\nContent.\n\n## Sub\n\nMore.\n")
        run_skeleton(str(src), str(out))
        skeleton_file = out / "doc.skeleton.json"
        assert skeleton_file.exists()
        data = json.loads(skeleton_file.read_text())
        assert isinstance(data, list)
        assert data[0]["title"] == "Title"

    def test_idempotent_overwrites_same_content(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        (src / "doc.md").write_text("# A\n\nContent.\n")
        run_skeleton(str(src), str(out))
        first = (out / "doc.skeleton.json").read_text()
        run_skeleton(str(src), str(out))
        second = (out / "doc.skeleton.json").read_text()
        assert first == second

    def test_raises_on_missing_input_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            run_skeleton(str(tmp_path / "nonexistent"), str(tmp_path / "out"))

    def test_no_md_files_logs_warning(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        # Should not raise, just log a warning
        run_skeleton(str(src), str(out))

    def test_preserves_relative_directory_tree(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        sub = src / "sub"
        sub.mkdir(parents=True)
        (sub / "doc.md").write_text("# Title\n\nContent.\n")
        out = tmp_path / "out"
        run_skeleton(str(src), str(out))
        assert (out / "sub" / "doc.skeleton.json").exists()
