from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import logging
from ..services.camera_analysis_service import camera_analyzer

router = APIRouter(prefix="/api/camera", tags=["camera"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraAnalysisRequest(BaseModel):
    image_data: str

@router.post("/analyze")
async def analyze_camera_frame(request: CameraAnalysisRequest) -> JSONResponse:
    """
    Analyze a camera frame for lighting, position, and stillness
    Expects base64 encoded image data
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_data.split(',')[1] if ',' in request.image_data else request.image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Use the camera analyzer service
        results = camera_analyzer.analyze_camera_frame(image)
        
        logger.info(f"Camera analysis completed successfully")
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in camera analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/reset")
async def reset_analysis() -> JSONResponse:
    """Reset the analysis state (useful for starting a new session)"""
    try:
        camera_analyzer.reset_stillness_detection()
        return JSONResponse(content={"message": "Analysis state reset successfully"})
    except Exception as e:
        logger.error(f"Error resetting analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

