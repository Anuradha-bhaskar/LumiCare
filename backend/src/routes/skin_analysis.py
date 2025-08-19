from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from typing import Dict, Any, List
import logging
from typing import Optional
from fastapi import Depends
from ..database.mongo import get_db
from ..services.skin_analyzer_main import skin_analyzer
from ..services.skin_persistence import save_skin_analysis
from ..services.skin_routine_service import generate_routine_for_user
from ..services.skin_diet_service import generate_diet_for_user

router = APIRouter(prefix="/api/skin", tags=["skin"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinAnalysisRequest(BaseModel):
    image_data: str
    clerk_user_id: Optional[str] = None


class RoutineRequest(BaseModel):
    clerk_user_id: str


class DietRequest(BaseModel):
    clerk_user_id: str


@router.post("/analyze")
async def analyze_skin(request: SkinAnalysisRequest, db=Depends(get_db)) -> JSONResponse:
    try:
        image_data = request.image_data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        height, width = image.shape[:2]
        if width < 200 or height < 200:
            raise HTTPException(status_code=400, detail="Image too small. Please use an image at least 200x200 pixels")
        results = skin_analyzer.analyze_skin(image)
        logger.info(f"Skin analysis completed successfully for image {width}x{height}")
        saved_to_db = False
        try:
            saved_to_db, _ = await save_skin_analysis(db, request.clerk_user_id, request.image_data, results)
        except Exception as save_err:
            logger.warning(f"Failed to save skin analysis to DB: {save_err}")
        return JSONResponse(content={**results, "saved_to_db": saved_to_db})
    except ValueError as e:
        logger.error(f"Invalid image data: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    except Exception as e:
        logger.error(f"Error in skin analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Skin analysis failed: {str(e)}")


@router.get("/history/{clerk_user_id}")
async def get_analysis_history(clerk_user_id: str, limit: int = 30, db=Depends(get_db)) -> JSONResponse:
    try:
        cursor = db.skin_analyses.find({"clerk_user_id": clerk_user_id}).sort("created_at", -1).limit(limit)
        items: List[Dict[str, Any]] = []
        async for doc in cursor:
            items.append({
                "id": str(doc.get("_id")),
                "timestamp": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                "image": doc.get("image"),
                "analysis": {
                    "metrics": doc.get("metrics", {}),
                    "comprehensive": doc.get("comprehensive", {}),
                    "skinHealth": doc.get("comprehensive", {}).get("skinHealth"),
                    "skinType": doc.get("comprehensive", {}).get("skinType"),
                    "concerns": doc.get("comprehensive", {}).get("concerns", []),
                }
            })
        return JSONResponse(content=items)
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analysis history")


@router.get("/routine/{clerk_user_id}")
async def get_latest_routine(clerk_user_id: str, db=Depends(get_db)) -> JSONResponse:
    try:
        doc = await db.skin_routines.find_one({"clerk_user_id": clerk_user_id}, sort=[("created_at", -1)])
        if not doc:
            return JSONResponse(content=None)
        serialized = {
            "id": str(doc.get("_id")),
            "clerk_user_id": doc.get("clerk_user_id"),
            "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
            "source_analysis_id": doc.get("source_analysis_id"),
            "routine": doc.get("routine")
        }
        return JSONResponse(content=serialized)
    except Exception as e:
        logger.error(f"Error fetching routine: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch routine")


@router.post("/routine")
async def generate_routine(payload: RoutineRequest, db=Depends(get_db)) -> JSONResponse:
    try:
        data = await generate_routine_for_user(db, payload.clerk_user_id)
        return JSONResponse(content=data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating routine: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate routine")


@router.get("/diet/{clerk_user_id}")
async def get_latest_diet(clerk_user_id: str, db=Depends(get_db)) -> JSONResponse:
    try:
        doc = await db.skin_diets.find_one({"clerk_user_id": clerk_user_id}, sort=[("created_at", -1)])
        if not doc:
            return JSONResponse(content=None)
        serialized = {
            "id": str(doc.get("_id")),
            "clerk_user_id": doc.get("clerk_user_id"),
            "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
            "source_analysis_id": doc.get("source_analysis_id"),
            "diet": doc.get("diet")
        }
        return JSONResponse(content=serialized)
    except Exception as e:
        logger.error(f"Error fetching diet: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch diet plan")


@router.post("/diet")
async def generate_diet(payload: DietRequest, db=Depends(get_db)) -> JSONResponse:
    try:
        data = await generate_diet_for_user(db, payload.clerk_user_id)
        return JSONResponse(content=data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating diet: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate diet plan")
