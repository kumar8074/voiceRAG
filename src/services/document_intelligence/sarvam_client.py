# ===================================================================================
# Project: VoiceRAG
# File: src/services/document_intelligence/sarvam_client.py
# Description: Async wrapper around Sarvam Document Intelligence API
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import asyncio
import io
import zipfile
import tempfile
import os
from time import time
from dotenv import load_dotenv
load_dotenv()

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from ...logger import logging
from ...config import SARVAM_DOC_INTEL_LANGUAGE, SARVAM_DOC_INTEL_OUTPUT_FORMAT

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# Sarvam job states that indicate a terminal outcome
_TERMINAL_STATES = {"Completed", "PartiallyCompleted", "Failed"}


class SarvamDocClient:
    """
    Thin async wrapper for Sarvam Document Intelligence.

    Each call to `process_pdf`:
        1. Creates a Sarvam DI job
        2. Uploads the PDF
        3. Starts the job
        4. Polls until terminal state (runs the blocking SDK call in a thread)
        5. Downloads the output ZIP in memory
        6. Extracts and concatenates all .md files inside the ZIP
        7. Returns the combined markdown string

    The underlying SDK is synchronous — all blocking calls are offloaded
    via asyncio.to_thread so we never block the event loop.
    """

    def __init__(
        self,
        api_key: str = SARVAM_API_KEY,
        language: str = SARVAM_DOC_INTEL_LANGUAGE,
        output_format: str = SARVAM_DOC_INTEL_OUTPUT_FORMAT,
    ):
        if not api_key:
            raise ValueError("SARVAM_API_KEY is not set.")
        self._api_key = api_key
        self.language = language
        self.output_format = output_format

    def _run_job_sync(self, file_path: str) -> bytes:
        """
        Blocking helper — runs the full Sarvam DI flow synchronously.
        Called via asyncio.to_thread so it never blocks the event loop.

        Returns the raw ZIP bytes of the processed output.
        """
    

        client = SarvamAI(api_subscription_key=self._api_key)

        logging.info(f"[DocIntel] Creating job for: {file_path}")
        job = client.document_intelligence.create_job(
            language=self.language,
            output_format=self.output_format,
        )

        logging.info(f"[DocIntel] Uploading file: {file_path}")
        job.upload_file(file_path)

        logging.info(f"[DocIntel] Starting job...")
        start_time = time()
        job.start()

        logging.info(f"[DocIntel] Waiting for completion...")
        status = job.wait_until_complete()
        end_time = time()
        logging.info(
            f"[DocIntel] Job finished with state: {status.job_state}, After: {(end_time - start_time):.2f}s"
        )

        if status.job_state == "Failed":
            raise RuntimeError(
                f"Sarvam Document Intelligence job failed for: {file_path}"
            )

        # SDK only accepts a file path string — write to a named tempfile,
        # read bytes back, then delete immediately so nothing lingers on disk.
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            job.download_output(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.remove(tmp_path)

    @staticmethod
    def _extract_markdown(zip_bytes: bytes) -> str:
        """
        Extract and concatenate all .md files from the ZIP bytes.
        Files are sorted by name so multi-page order is preserved.
        """
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            md_files = sorted(
                [name for name in zf.namelist() if name.endswith(".md")]
            )

            if not md_files:
                raise ValueError(
                    "Sarvam returned a ZIP with no .md files. "
                    "Check output_format='md' in your job config."
                )

            parts = []
            for name in md_files:
                content = zf.read(name).decode("utf-8", errors="replace")
                parts.append(content)

        combined = "\n\n".join(parts)
        logging.info(
            f"[DocIntel] Extracted {len(md_files)} .md file(s), "
            f"{len(combined)} total chars."
        )
        return combined

    async def process_pdf(self, file_path: str) -> str:
        """
        Async entry point. Submits the PDF to Sarvam Document Intelligence
        and returns the extracted markdown as a plain string.

        Args:
            file_path: Absolute or relative path to the PDF on disk.

        Returns:
            Markdown string representing the document's content.

        Raises:
            RuntimeError: If the Sarvam job fails.
            ApiError:     If the Sarvam API returns an error response.
        """
        try:
            zip_bytes = await asyncio.to_thread(self._run_job_sync, file_path)
            return self._extract_markdown(zip_bytes)

        except ApiError as e:
            logging.error(
                f"[DocIntel] Sarvam API error (HTTP {e.status_code}) "
                f"for {file_path}: {e.body}"
            )
            raise RuntimeError(
                f"Sarvam API error {e.status_code} for {file_path}: {e.body}"
            ) from e

        except Exception as e:
            logging.error(f"[DocIntel] Unexpected error for {file_path}: {e}")
            raise
        
# Example usage:
if __name__ == "__main__":
    import asyncio
    client=SarvamDocClient()
    result=asyncio.run(client.process_pdf("tmp/Springer_Nature_LaTeX_Template.pdf"))
    logging.info(f"Extracted content: {result}")