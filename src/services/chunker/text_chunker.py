# ===================================================================================
# Project: VoiceRAG
# File: src/services/chunker/text_chunker.py
# Description: Splits text documents into smaller chunks for better processing.
# Author: LALAN KUMAR
# Created: [02-03-2026]
# Updated: [03-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

from ...config import CHUNKER_CHUNK_SIZE, CHUNKER_CHUNK_OVERLAP

class TextChunker:
    def __init__(self,chunk_size=CHUNKER_CHUNK_SIZE, chunk_overlap=CHUNKER_CHUNK_OVERLAP):
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split(self,documents:List[Document])->List[Document]:
        return self.splitter.split_documents(documents)
    
    
# Example usage:
if __name__ == "__main__":
    from ..pdf_parser.parser import PDFParser
    docs=PDFParser.parse_pdf("tmp/83577772-b923-4c40-891d-2edd3fe26fad_Startup India Kit_v5.pdf")
    chunker=TextChunker()
    chunks=chunker.split(docs)
    print(f"Original document count: {len(docs)}")
    print(f"Chunked document count: {len(chunks)}")
    print(f"Type of first chunk: {type(chunks[0])}")
    print(f"First chunk content: {chunks[0].page_content[:500]}")