# ===================================================================================
# Project: VoiceRAG
# File: src/services/pdf_parser/parser.py
# Description: Parses PDF files and returns list of documents.
# Author: LALAN KUMAR
# Created: [02-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

class PDFParser:
    @staticmethod
    async def parse_pdf(file_path:str)->list[Document]:
        """Parse a PDF file and return its content as a list of documents."""
        try:
            loader = PyMuPDFLoader(file_path)
            documents = await loader.aload()
            return documents
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF: {e}")
  
  
# Example usage:
if __name__ == "__main__":
    import asyncio
    from ...logger import logging
    
    docs = asyncio.run(PDFParser.parse_pdf("tmp/8dffad42-d686-4691-a1b5-7ffa0b9a9880_RLAlgsInMDPs.pdf"))
    logging.info("PDF Parsing completed")
    logging.info(f"Loaded content {docs[0].page_content[:500]}")  # Print the first 500 characters of the first document's content


