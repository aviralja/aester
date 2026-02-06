"""
FastAPI Application for Civic Issue Detection and AI Image Checking

This API uses the functions from civic_functions.py module to provide
REST endpoints for image analysis.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os

# Import our custom functions
from civic_functions import (
    detect_civic_issue_from_bytes,
    check_if_ai_generated_bytes,
    CivicIssue
)


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Civic Issue Detection API",
    description="API for detecting civic issues from images and checking if images are AI-generated",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC RESPONSE MODELS
# ============================================================================

class CivicIssueResponse(BaseModel):
    """Response model for civic issue detection"""
    success: bool
    data: CivicIssue
    message: str = ""


class AIGeneratedResponse(BaseModel):
    """Response model for AI image detection"""
    success: bool
    is_ai_generated: bool
    confidence: float
    full_result: Dict[Any, Any] = {}
    message: str = ""


class CompleteAnalysisResponse(BaseModel):
    """Response model for complete analysis"""
    success: bool
    civic_issue: Dict[str, str]
    ai_detection: Dict[str, Any]
    message: str = ""


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def validate_image_file(file: UploadFile) -> bytes:
    """
    Validate that the uploaded file is an image and return its bytes.
    
    Args:
        file: Uploaded file from FastAPI
        
    Returns:
        bytes: Image file bytes
        
    Raises:
        HTTPException: If file is not an image
    """
    # Check content type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected image, got {file.content_type}"
        )
    
    # Check file size (max 10MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB"
        )
    
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )
    
    return content


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - API information
    
    Returns basic information about the API and available endpoints.
    """
    return {
        "message": "Civic Issue Detection API",
        "version": "2.0.0",
        "description": "Detect civic issues and check AI-generated images",
        "endpoints": {
            "/detect-civic-issue": "POST - Detect civic issues from image",
            "/check-ai-generated": "POST - Check if image is AI-generated",
            "/analyze-complete": "POST - Complete analysis (civic + AI)",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation",
            "/redoc": "GET - Alternative API documentation"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the API and configuration status.
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "api_configured": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "sightengine": bool(os.getenv("SIGHTENGINE_API_USER") and os.getenv("SIGHTENGINE_API_SECRET"))
        }
    }


@app.post(
    "/detect-civic-issue",
    response_model=CivicIssueResponse,
    responses={
        200: {"description": "Successful civic issue detection"},
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Detect Civic Issues",
    tags=["Civic Issue Detection"]
)
async def detect_civic_issue_endpoint(file: UploadFile = File(..., description="Image file to analyze")):
    """
    Detect civic issues from an uploaded image.
    
    This endpoint analyzes an image using AI to identify various civic problems such as:
    - Potholes and road damage
    - Garbage overflow
    - Water leakage
    - Broken infrastructure
    - And many more (see documentation)
    
    **Parameters:**
    - **file**: Image file (JPEG, PNG, etc.) - Maximum 10MB
    
    **Returns:**
    - **success**: Whether the detection was successful
    - **data**: Object containing issue_type and description
    - **message**: Status message
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "issue_type": "Potholes",
            "description": "Large pothole on main road causing traffic issues"
        },
        "message": "Civic issue detection completed successfully"
    }
    ```
    """
    try:
        # Validate and read image
        image_bytes = await validate_image_file(file)
        
        # Detect civic issue using our function
        civic_issue = detect_civic_issue_from_bytes(image_bytes)
        
        return CivicIssueResponse(
            success=True,
            data=civic_issue,
            message="Civic issue detection completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post(
    "/check-ai-generated",
    response_model=AIGeneratedResponse,
    responses={
        200: {"description": "Successful AI detection"},
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Check if Image is AI-Generated",
    tags=["AI Detection"]
)
async def check_ai_generated_endpoint(
    file: UploadFile = File(..., description="Image file to check"),
    threshold: float = 0.5
):
    """
    Check if an uploaded image is AI-generated.
    
    This endpoint uses Sightengine's AI detection model to determine whether
    an image was created by AI or is a real photograph.
    
    **Parameters:**
    - **file**: Image file (JPEG, PNG, etc.) - Maximum 10MB
    - **threshold**: Confidence threshold (0.0 to 1.0). Default: 0.5
      - Values above threshold are considered AI-generated
      - Lower threshold = more sensitive (more false positives)
      - Higher threshold = more strict (fewer false positives)
    
    **Returns:**
    - **success**: Whether the detection was successful
    - **is_ai_generated**: Boolean indicating if image is AI-generated
    - **confidence**: Confidence score (0.0 to 1.0)
    - **full_result**: Complete API response with additional details
    - **message**: Status message
    
    **Example Response:**
    ```json
    {
        "success": true,
        "is_ai_generated": false,
        "confidence": 0.23,
        "full_result": {
            "status": "success",
            "type": {
                "ai_generated": 0.23
            }
        },
        "message": "AI detection completed successfully"
    }
    ```
    """
    try:
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="Threshold must be between 0.0 and 1.0"
            )
        
        # Validate and read image
        image_bytes = await validate_image_file(file)
        
        # Check if AI-generated using our function
        is_ai, full_result = check_if_ai_generated_bytes(image_bytes, threshold=threshold)
        
        # Extract confidence score
        confidence = full_result.get("type", {}).get("ai_generated", 0) if full_result.get("status") == "success" else 0
        
        return AIGeneratedResponse(
            success=full_result.get("status") == "success",
            is_ai_generated=is_ai,
            confidence=confidence,
            full_result=full_result,
            message="AI detection completed successfully" if full_result.get("status") == "success" else "AI detection failed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post(
    "/analyze-complete",
    response_model=CompleteAnalysisResponse,
    responses={
        200: {"description": "Successful complete analysis"},
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Complete Analysis (Civic + AI)",
    tags=["Complete Analysis"]
)
async def analyze_complete_endpoint(
    file: UploadFile = File(..., description="Image file to analyze"),
    ai_threshold: float = 0.5
):
    
    try:
        # Validate threshold
        if not 0.0 <= ai_threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="AI threshold must be between 0.0 and 1.0"
            )
        
        # Validate and read image
        image_bytes = await validate_image_file(file)
        
        # Detect civic issue
        civic_issue = detect_civic_issue_from_bytes(image_bytes)
        
        # Check if AI-generated
        is_ai, ai_result = check_if_ai_generated_bytes(image_bytes, threshold=ai_threshold)
        
        # Extract confidence score
        confidence = ai_result.get("type", {}).get("ai_generated", 0) if ai_result.get("status") == "success" else 0
        
        return CompleteAnalysisResponse(
            success=True,
            civic_issue={
                "issue_type": civic_issue.issue_type,
                "description": civic_issue.description
            },
            ai_detection={
                "is_ai_generated": is_ai,
                "confidence": confidence,
                "status": ai_result.get("status", "unknown")
            },
            message="Complete analysis finished successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# ============================================================================
# ADDITIONAL UTILITY ENDPOINTS
# ============================================================================

@app.get(
    "/supported-issues",
    summary="Get Supported Issue Types",
    tags=["Information"]
)
async def get_supported_issues():
    """
    Get a list of all supported civic issue types.
    
    Returns all the civic issue categories and specific types that the
    detection model can identify.
    """
    return {
        "categories": {
            "Road & Transportation": [
                "Potholes",
                "Broken roads",
                "Damaged footpaths",
                "Speed breaker damage",
                "Road waterlogging",
                "Traffic signal not working",
                "Missing signboards",
                "Bus stop shelter damage"
            ],
            "Sanitation & Waste Management": [
                "Garbage overflow",
                "Missed garbage collection",
                "Littering on streets",
                "Open dumping",
                "Public toilet dirty",
                "Dead animal on road",
                "Drain cleaning needed"
            ],
            "Water Supply & Sewerage": [
                "Water pipe leakage",
                "No water supply",
                "Low water pressure",
                "Sewer overflow",
                "Open manhole",
                "Blocked drainage",
                "Contaminated water"
            ],
            "Electricity & Street Lighting": [
                "Streetlight not working",
                "Power outage",
                "Fallen electric pole",
                "Exposed wires",
                "Transformer issue"
            ],
            "Public Health & Safety": [
                "Stray animals",
                "Mosquito breeding",
                "Unsafe building",
                "Fire hazard",
                "Chemical spill",
                "Gas leak",
                "Accident prone area"
            ],
            "Environment & Parks": [
                "Tree fallen",
                "Illegal tree cutting",
                "Park maintenance issue",
                "Air pollution",
                "Water pollution",
                "Noise pollution"
            ],
            "Traffic & Law Enforcement": [
                "Illegal parking",
                "Signal jumping",
                "Wrong side driving",
                "Encroachment on road",
                "Overloaded vehicles"
            ],
            "Public Infrastructure": [
                "Broken benches",
                "Damaged playground equipment",
                "Bus stop damaged",
                "Broken railing",
                "Public building maintenance"
            ]
        }
    }


# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    print("=" * 70)
    print("🚀 Starting Civic Issue Detection API")
    print("=" * 70)
    print(f"📍 Server: http://{HOST}:{PORT}")
    print(f"📚 Documentation: http://{HOST}:{PORT}/docs")
    print(f"📖 Alternative Docs: http://{HOST}:{PORT}/redoc")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )