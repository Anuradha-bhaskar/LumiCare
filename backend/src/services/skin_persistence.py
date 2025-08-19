from typing import Any, Dict, Optional, Tuple
from datetime import datetime
import logging

from ..models.analysis import SkinAnalysisDoc, SkinMetrics, SkinComprehensive

logger = logging.getLogger(__name__)


async def save_skin_analysis(db, clerk_user_id: Optional[str], original_image_data: str, results: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        if not clerk_user_id:
            return False, None
        metrics_model = SkinMetrics.model_validate(results.get("metrics", {}))
        comprehensive_model = SkinComprehensive.model_validate(results.get("comprehensive", {}))
        doc_model = SkinAnalysisDoc(
            clerk_user_id=clerk_user_id,
            created_at=datetime.utcnow(),
            image=original_image_data,
            metrics=metrics_model,
            comprehensive=comprehensive_model,
            face_region=results.get("face_region", "")
        )
        insert_res = await db.skin_analyses.insert_one(doc_model.model_dump(by_alias=True))
        return insert_res.inserted_id is not None, str(insert_res.inserted_id) if insert_res.inserted_id else None
    except Exception as e:
        logger.warning(f"Failed to save skin analysis to DB: {e}")
        return False, None 