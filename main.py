"""
Minimal FastAPI Application for Civic Issue Detection

This simplified API matches the updated `civic_functions.py` model.

Endpoints kept:
- GET /health
- POST /detect-civic-issue  -> returns `CivicIssue` from bytes
- POST /analyze-complete    -> returns combined civic issue + AI check

Removed other endpoints as requested.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os

from civic_functions import (
    detect_civic_issue_from_bytes,
    detect_civic_issue_from_bytes_description,
    check_if_ai_generated_bytes,
    CivicIssue,
)


app = FastAPI(
    title="Civic Issue Detection API (minimal)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CivicIssueResponse(BaseModel):
    success: bool
    data: CivicIssue
    message: str = ""


class CompleteAnalysisResponse(BaseModel):
    success: bool
    civic_issue: Dict[str, Any]
    ai_detection: Dict[str, Any]
    message: str = ""


async def validate_image_file(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Expected image, got {file.content_type}")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")

    return content


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "api_configured": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "sightengine": bool(os.getenv("SIGHTENGINE_API_USER") and os.getenv("SIGHTENGINE_API_SECRET")),
        },
    }


@app.post("/detect-civic-issue", response_model=CivicIssueResponse)
async def detect_civic_issue_endpoint(file: UploadFile = File(..., description="Image file to analyze")):
    try:
        image_bytes = await validate_image_file(file)
        civic_issue = detect_civic_issue_from_bytes(image_bytes)

        return CivicIssueResponse(success=True, data=civic_issue, message="Civic issue detection completed")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.post("/detect-civic-issue-with-description", response_model=CivicIssueResponse)
async def detect_civic_issue_with_description(
    file: UploadFile = File(..., description="Image file to analyze"),
    description: str = Form(..., description="User-provided description to verify with the image"),
):
    try:
        image_bytes = await validate_image_file(file)

        civic_issue = detect_civic_issue_from_bytes_description(image_bytes, description)

        return CivicIssueResponse(success=True, data=civic_issue, message="Civic issue detection with description completed")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.post("/analyze-complete", response_model=CompleteAnalysisResponse)
async def analyze_complete_endpoint(file: UploadFile = File(..., description="Image file to analyze"), ai_threshold: float = 0.5):
    try:
        if not 0.0 <= ai_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="AI threshold must be between 0.0 and 1.0")

        image_bytes = await validate_image_file(file)

        # Detect civic issue from bytes
        civic_issue = detect_civic_issue_from_bytes(image_bytes)

        # Check if AI-generated using bytes helper
        is_ai, ai_result = check_if_ai_generated_bytes(image_bytes, threshold=ai_threshold)

        confidence = ai_result.get("type", {}).get("ai_generated", 0) if ai_result.get("status") == "success" else 0

        civic_issue_dict = {
            "valid": getattr(civic_issue, "valid", None),
            "justification": getattr(civic_issue, "justification", None),
            "description": getattr(civic_issue, "description", None),
            "severity": getattr(civic_issue, "severity", None),
        }

        ai_detection = {
            "is_ai_generated": is_ai,
            "confidence": confidence,
            "status": ai_result.get("status", "unknown"),
        }

        return CompleteAnalysisResponse(
            success=not ai_detection["is_ai_generated"] and civic_issue_dict["valid"],
            civic_issue=civic_issue_dict,
            ai_detection=ai_detection,
            message="Complete analysis finished",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.post("/analyze-complete-description", response_model=CompleteAnalysisResponse)
async def analyze_complete_endpoint_description(
    file: UploadFile = File(..., description="Image file to analyze"),
    description: str = Form(..., description="User-provided description to verify with the image"),
    ai_threshold: float = 0.5,
):
    try:
        if not 0.0 <= ai_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="AI threshold must be between 0.0 and 1.0")

        image_bytes = await validate_image_file(file)

        # Use the description-aware detector
        civic_issue = detect_civic_issue_from_bytes_description(image_bytes, description)

        # Check if AI-generated using bytes helper
        is_ai, ai_result = check_if_ai_generated_bytes(image_bytes, threshold=ai_threshold)

        confidence = ai_result.get("type", {}).get("ai_generated", 0) if ai_result.get("status") == "success" else 0

        civic_issue_dict = {
            "valid": getattr(civic_issue, "valid", None),
            "justification": getattr(civic_issue, "justification", None),
            "description": getattr(civic_issue, "description", None),
            "severity": getattr(civic_issue, "severity", None),
        }

        ai_detection = {
            "is_ai_generated": is_ai,
            "confidence": confidence,
            "status": ai_result.get("status", "unknown"),
        }

        return CompleteAnalysisResponse(
            success=not ai_detection["is_ai_generated"] and civic_issue_dict["valid"],
            civic_issue=civic_issue_dict,
            ai_detection=ai_detection,
            message="Complete analysis with description finished",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


if __name__ == "__main__":
    import uvicorn

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 3000))

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
