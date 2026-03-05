# ===================================================================================
# Project: VoiceRAG
# File: src/services/document_intelligence/md_chunker.py
# Description: Structure-aware Markdown chunker
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...config import CHUNKER_CHUNK_SIZE, CHUNKER_CHUNK_OVERLAP
from ...logger import logging

# Matches any ATX heading line: # / ## / ### / etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownChunker:
    """
    Structure-aware chunker for Markdown text produced by Sarvam Document Intelligence.

    Strategy:
        1. Split the full markdown on heading boundaries → logical sections.
           Each section = heading line + its body text until the next heading.
        2. Any section that fits within chunk_size is kept as-is (one Document).
        3. Any section that exceeds chunk_size is further split with
           RecursiveCharacterTextSplitter so we never produce oversized chunks.

    This preserves semantic coherence: chunks don't straddle two topics,
    tables stay with their heading, and paragraph flow is respected.
    """

    def __init__(
        self,
        chunk_size: int = CHUNKER_CHUNK_SIZE,
        chunk_overlap: int = CHUNKER_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Fallback splitter for oversized sections
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _split_into_sections(self, markdown: str) -> List[str]:
        """
        Split markdown into logical sections on heading boundaries.

        Each section includes its heading line and all content up to
        (but not including) the next heading. A leading block of text
        before the first heading is kept as its own section.
        """
        # Find all heading positions
        heading_positions = [m.start() for m in _HEADING_RE.finditer(markdown)]

        if not heading_positions:
            # No headings at all — treat entire document as one section
            return [markdown.strip()] if markdown.strip() else []

        sections: List[str] = []

        # Text before the very first heading (e.g. abstract / preamble)
        preamble = markdown[: heading_positions[0]].strip()
        if preamble:
            sections.append(preamble)

        # Slice between consecutive heading positions
        for i, start in enumerate(heading_positions):
            end = heading_positions[i + 1] if i + 1 < len(heading_positions) else len(markdown)
            section = markdown[start:end].strip()
            if section:
                sections.append(section)

        return sections

    def split(self, markdown: str, metadata: dict | None = None) -> List[Document]:
        """
        Split a markdown string into a list of LangChain Documents.

        Args:
            markdown: Raw markdown text from Sarvam Document Intelligence.
            metadata: Optional dict merged into every chunk's metadata
                      (e.g. user_id, doc_id, filename).

        Returns:
            List[Document] — each with page_content set to the chunk text
            and metadata copied from the provided dict.
        """
        if not markdown or not markdown.strip():
            logging.warning("[MdChunker] Empty markdown received — returning no chunks.")
            return []

        base_meta = metadata or {}
        sections = self._split_into_sections(markdown)
        logging.info(f"[MdChunker] {len(sections)} logical section(s) identified.")

        chunks: List[Document] = []

        for section in sections:
            if len(section) <= self.chunk_size:
                # Section fits — keep it whole
                chunks.append(Document(page_content=section, metadata=dict(base_meta)))
            else:
                # Section too large — fall back to size-based splitting
                sub_docs = self._fallback_splitter.create_documents(
                    texts=[section],
                    metadatas=[dict(base_meta)],
                )
                chunks.extend(sub_docs)

        logging.info(
            f"[MdChunker] Produced {len(chunks)} chunk(s) "
            f"from {len(sections)} section(s)."
        )
        def _is_valid_chunk(text: str) -> bool:
            stripped = text.strip()
            # Too short to be useful
            if len(stripped) < 30:
                return False
            # Bare heading only (e.g. "## Abstract" with nothing else)
            if re.match(r'^#{1,6}\s+\S.*$', stripped) and '\n' not in stripped:
                return False
            # Base64 blob — high ratio of base64 chars, no spaces
            non_b64 = re.sub(r'[A-Za-z0-9+/=]', '', stripped)
            if len(stripped) > 50 and len(non_b64) / len(stripped) < 0.05:
                return False
            return True
        
        chunks = [c for c in chunks if _is_valid_chunk(c.page_content)]
        logging.info(f"[MdChunker] {len(chunks)} chunk(s) after junk filtering")
        
        return chunks

   
    
# Example usage:
if __name__ == "__main__":
    import asyncio
    from .sarvam_client import SarvamDocClient
    from time import time
    
    client=SarvamDocClient()
    result=asyncio.run(client.process_pdf("tmp/Springer_Nature_LaTeX_Template.pdf"))
    logging.info(f"Extracted content: {result}")
    
    chunker=MarkdownChunker()
    logging.info("Starting chunking...")
    start_time=time()
    chunks=chunker.split(result)
    end_time=time()
    logging.info(f"Chunking completed after: {(end_time-start_time):.2f}s")
    logging.info(f"Extracted Chunks: {chunks}")