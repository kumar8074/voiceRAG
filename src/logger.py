# ===================================================================================
# Project: VoiceRAG
# File: src/logger.py
# Description: Used for logging
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import logging
import os
import sys
from datetime import datetime

# Define log filename format using current timestamp
LOG_FILENAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory path for logs
LOGS_DIR = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the full path to the log file
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILENAME)

# Configure basic logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)