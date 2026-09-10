#!/bin/bash
set -e

echo "[*] Starting FastAPI application on port 7860..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
