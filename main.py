from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uvicorn

from src.routers.file_upload import router as file_upload

app = FastAPI()
app.include_router(file_upload)

# Ensure upload directory exists
os.makedirs("tmp", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    """Serve the chat UI at the root URL."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
    )
