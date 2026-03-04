# ===================================================================================
# Project: VoiceRAG
# File: src/routers/file_upload.py
# Description: File upload router 
# Author: LALAN KUMAR
# Created: [02-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from typing import List
import aiofiles
import os
from uuid import uuid4

from ..services.indexing.indexer import QdrantIndexer
from ..logger import logging

router = APIRouter(prefix="/api/v1", tags=["file_upload"])

UPLOAD_DIR = "tmp"
MAX_FILES = 3
MAX_FILE_SIZE = 50*1024*1024  # 50 MB

# Singleton indexer — reuses shared Qdrant + embedding singletons
_indexer = QdrantIndexer()

_jobs: dict = {} # In memory job status

async def _run_indexing(job_id: str, files_to_index: list[dict], user_id: str, session_id: str):
    """
    Background task: index all uploaded files, update job status when done.
    files_to_index: list of { file_path, original_name }
    """
    try:
        doc_ids=[]
        for f in files_to_index:
            logging.info(f"Indexing {f['original_name']} for user={user_id}")
            result=await _indexer.index_pdf(
                file_path=f["file_path"],
                user_id=user_id,
                session_id=session_id,
                language="auto"
            )
            doc_ids.append({
                "doc_id": result["doc_id"],
                "filename": f["original_name"]
            })
            logging.info(f"Indexed {f['original_name']}, doc_id={result['doc_id']}")

        _jobs[job_id] = {
            "status": "done", 
            "detail": "All files indexed successfully.",
            "docs": doc_ids
        }
        logging.info(f"Job {job_id} complete: {doc_ids}")

    except Exception as e:
        _jobs[job_id] = {"status": "error", "detail": str(e), "docs": []}
        logging.error(f"Job {job_id} failed: {e}")


@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile]=File(...),
    user_id: str = Form(...),
    session_id: str = Form(...)
):
    
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required for indexing.")
    
    if len(files)> MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files allowed per upload.")
    
    saved_files =[]
    files_to_index=[]
    
    for file in files:
        filename=f"{uuid4()}_{os.path.basename(file.filename)}" # Ensure only filename is used, prevent traversal attacks
        file_path=os.path.join(UPLOAD_DIR, filename)
        
        size=0
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            while True:
                chunk=await file.read(1024*1024) # Read in 1 MB chunks
                if not chunk:
                    break
                
                size+=len(chunk)
                
                if size>MAX_FILE_SIZE:
                    # Cleanup partial file, if any
                    await out_file.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=400,
                        detail=f"{filename} exceeds size limit of {MAX_FILE_SIZE} MB"
                    )
                await out_file.write(chunk)
                
        saved_files.append(filename)
        files_to_index.append({"file_path": file_path, "original_name": filename})
        
        job_id = str(uuid4())
        _jobs[job_id] = {"status": "indexing", "detail": "Indexing in progress...", "docs":[]} 
        
        # Kick off indexing in the background — upload response returns immediately
        background_tasks.add_task(_run_indexing, job_id, files_to_index, user_id, session_id)
        logging.info(f"Upload complete, indexing job {job_id} started for user={user_id}")
    
    return {
        "job_id": job_id,
        "files": saved_files
    }
    
@router.get("/index-status/{job_id}")
async def index_status(job_id: str):
    """
    Poll this endpoint to check indexing progress.
    Returns: { status: 'indexing' | 'done' | 'error', detail: str }
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str, user_id: str):
    """
    Delete all Qdrant vectors for a specific document belonging to a user.
    """
    if not user_id or not doc_id:
        raise HTTPException(status_code=400, detail="user_id and doc_id are required.")

    try:
        status = _indexer.delete_document(user_id=user_id, doc_id=doc_id)
        logging.info(f"Deleted doc_id={doc_id} for user={user_id}, status={status}")
        return {"status": "deleted", "doc_id": doc_id}
    except Exception as e:
        logging.error(f"Failed to delete doc_id={doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")