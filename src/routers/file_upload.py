from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import aiofiles
import os
from uuid import uuid4

router = APIRouter(prefix="/api/v1", tags=["file_upload"])

UPLOAD_DIR = "tmp"
MAX_FILES = 3
MAX_FILE_SIZE = 5*1024*1024  # 5 MB

@router.post("/upload")
async def upload_files(files: List[UploadFile]=File(...)):
    if len(files)> MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files allowed per upload.")
    
    saved_files =[]
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
                    raise HTTPException(
                        status_code=400,
                        detail=f"{filename} exceeds size limit of {MAX_FILE_SIZE} MB"
                    )
                await out_file.write(chunk)
        saved_files.append(filename)
    
    return {
        "status": "Upload successful",
        "files": saved_files
    }