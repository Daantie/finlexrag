"""Proxy-Pointer RAG — data contracts for the offline indexing pipeline.

See docs/requirements.md → Data models / contracts.
"""

import hashlib

from pydantic import BaseModel, Field


class SkeletonNode(BaseModel):
    """A single node in the heading tree of a parsed Markdown document."""

    node_id: str = Field(description="Stable hash, e.g. sha1(doc_id|path)[:12]")
    title: str
    level: int = Field(description="Heading level (1..6)")
    path: list[str] = Field(description="Hierarchical path of titles from root to this node")
    char_start: int = Field(description="Byte offset of the heading line in the source .md")
    char_end: int = Field(description="Byte offset of the last character of this section")
    children: list["SkeletonNode"] = Field(default_factory=list)

    @staticmethod
    def make_node_id(doc_id: str, path: list[str]) -> str:
        """Return a 12-char hex node_id stable across reruns."""
        raw = doc_id + "|" + "|".join(path)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


class Chunk(BaseModel):
    """A text chunk produced by the section-aware chunker."""

    doc_id: str
    node_id: str
    hierarchical_path: list[str]
    source_path: str = Field(description="Relative path of the .md file")
    char_start: int
    char_end: int
    text: str


class QdrantPayload(BaseModel):
    """Payload stored alongside each Qdrant vector point."""

    doc_id: str
    node_id: str
    hierarchical_path: list[str]
    source_path: str
    char_start: int
    char_end: int
    text_preview: str = Field(description="First ~256 chars of the chunk text, for debug only")
