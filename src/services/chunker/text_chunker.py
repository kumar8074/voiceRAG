# ===================================================================================
# Project: VoiceRAG
# File: src/services/chunker/text_chunker.py
# Description: Splits text documents into smaller chunks for better processing.
# Author: LALAN KUMAR
# Created: [03-03-2026]
# Updated: [03-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class TextChunker:
    def __init__(self,chunk_size=800, chunk_overlap=100):
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split(self,documents:List[Document])->List[Document]:
        return self.splitter.split_documents(documents)