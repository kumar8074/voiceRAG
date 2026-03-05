# ===================================================================================
# Project: VoiceRAG
# File: src/services/chunker/text_chunker.py
# Description: Splits text documents into smaller chunks for better processing (Marked Redundant)
# Author: LALAN KUMAR
# Created: [02-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

# No Longer Needed. 
# Logic is Now Handled by `src/services/document_intelligence/md_chunker.py` 
# for document intelligence capabilities

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
    from ...logger import logging
    from time import time
    
    logging.info("Initiating PDF Parsing...")
    start_time=time()
    docs=PDFParser.parse_pdf("tmp/83577772-b923-4c40-891d-2edd3fe26fad_Startup India Kit_v5.pdf")
    end_time=time()
    logging.info(f"Parsing Completed in {(end_time-start_time):.2f} seconds")
    
    logging.info("Initiating Document Chunking...")
    start_time=time()
    chunker=TextChunker()
    chunks=chunker.split(docs)
    end_time=time()
    logging.info(f"Chunking Comleted in {(end_time-start_time):.2f}seconds")
    
    logging.info(f"Original document count: {len(docs)}")
    logging.info(f"Chunked document count: {len(chunks)}")
    logging.info(f"Type of first chunk: {type(chunks[0])}")
    logging.info(f"First chunk content: {chunks[0].page_content[:500]}")