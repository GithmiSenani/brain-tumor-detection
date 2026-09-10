import logging
import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app.main")

from app.auth.auth_handler import get_current_user, get_optional_user, User
from app.validation.file_validator import validate_uploaded_image
from app.inference.classifier import classifier
from app.records import records_handler

app = FastAPI(
    title="Brain Tumor Detection API",
    description="Backend service for brain MRI tumor prediction and patient records storage.",
    version="1.0.0"
)

# CORS configuration
origins = [
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",  # Common frontend port
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory IP rate limiter for predictions to prevent abuse
prediction_rate_limit = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
MAX_PREDICTIONS_PER_WINDOW = 10

def enforce_rate_limit(ip: str):
    now = time.time()
    # Filter out timestamps older than the window
    prediction_rate_limit[ip] = [t for t in prediction_rate_limit[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(prediction_rate_limit[ip]) >= MAX_PREDICTIONS_PER_WINDOW:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {MAX_PREDICTIONS_PER_WINDOW} predictions per minute. Please try again later."
        )
    prediction_rate_limit[ip].append(now)

@app.get("/api/health")
def health_check():
    model_status = classifier.get_model_status()
    return {
        "status": "healthy",
        "supabase_connected": records_handler.is_supabase_enabled(),
        "model_loaded": not classifier.is_mock,
        "model_version": classifier.model_version,
        "model_pipeline": model_status
    }

@app.post("/api/predict")
async def predict_tumor(
    request: Request,
    file: UploadFile = File(...),
    user: Optional[User] = Depends(get_optional_user)
):
    """
    Accepts an MRI scan image and performs prediction.
    If authenticated (user is not None), saves the scan and prediction details to Supabase.
    """
    # 1. Rate Limit Check
    client_ip = request.client.host if request.client else "unknown-ip"
    enforce_rate_limit(client_ip)
    
    # 2. Server-side file validation (size, format, corruption check)
    file_bytes = await validate_uploaded_image(file)
    
    # 3. Model Inference
    try:
        prediction_result = classifier.predict(file_bytes, filename=file.filename or "mri_scan.png")
        label = prediction_result["prediction_label"]
        confidence = prediction_result["confidence"]
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error running prediction model."
        )

    response_data = {
        "prediction_label": label,
        "confidence": confidence,
        "model_version": classifier.model_version,
        "saved_to_history": False,
        "segmented_pixels": prediction_result.get("segmented_pixels", 0),
        "explanation_text": prediction_result.get("explanation_text", ""),
        "images": prediction_result.get("images", {})
    }

    # 4. Save to history if logged in
    if user:
        try:
            record = await records_handler.save_scan_record(
                user_id=user.id,
                file_bytes=file_bytes,
                filename=file.filename or "mri_scan.png",
                content_type=file.content_type or "image/png",
                prediction_label=label,
                confidence=confidence,
                model_version=classifier.model_version
            )
            response_data["saved_to_history"] = True
            response_data["record"] = record
        except Exception as e:
            logger.error(f"Failed to save record for user {user.id}: {e}")
            # We don't fail the prediction request if the database save fails, 
            # but we inform the client that it wasn't saved.
            response_data["save_error"] = "Prediction succeeded but failed to save to history."
            
    return response_data

@app.get("/api/history")
async def get_history(user: User = Depends(get_current_user)):
    """Retrieves the history of scan records for the authenticated user."""
    try:
        records = await records_handler.get_user_history(user.id)
        return {"records": records}
    except Exception as e:
        logger.error(f"Failed to fetch history for user {user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve history records."
        )

@app.get("/api/history/{record_id}")
async def get_history_detail(record_id: str, user: User = Depends(get_current_user)):
    """Retrieves details for a specific scan record."""
    try:
        record = await records_handler.get_scan_record(user.id, record_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan record not found or access denied."
            )
        return record
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch record {record_id} for user {user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving scan record."
        )

@app.delete("/api/history/{record_id}")
async def delete_history_record(record_id: str, user: User = Depends(get_current_user)):
    """Deletes a scan record and its file from history."""
    try:
        success = await records_handler.delete_scan_record(user.id, record_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan record not found or access denied."
            )
        return {"detail": "Record deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete record {record_id} for user {user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting scan record."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
