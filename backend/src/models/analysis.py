from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class AcneMetric(BaseModel):
    severity: Optional[str] = None
    count: Optional[int] = None
    score: Optional[float] = None
    avg_size: Optional[float] = None
    description: Optional[str] = None


class OilinessMetric(BaseModel):
    severity: Optional[str] = None
    score: Optional[float] = None
    t_zone_brightness: Optional[float] = None
    texture_variance: Optional[float] = None
    description: Optional[str] = None


class PigmentationMetric(BaseModel):
    severity: Optional[str] = None
    score: Optional[float] = None
    color_variation: Optional[float] = None
    dark_spots_percentage: Optional[float] = None
    description: Optional[str] = None


class WrinklesMetric(BaseModel):
    severity: Optional[str] = None
    score: Optional[float] = None
    density: Optional[float] = None
    line_count: Optional[int] = None
    description: Optional[str] = None


class PoresMetric(BaseModel):
    severity: Optional[str] = None
    count: Optional[int] = None
    avg_size: Optional[float] = None
    density: Optional[float] = None
    score: Optional[float] = None
    description: Optional[str] = None


class HydrationMetric(BaseModel):
    severity: Optional[str] = None
    score: Optional[float] = None
    smoothness: Optional[float] = None
    texture_uniformity: Optional[float] = None
    description: Optional[str] = None


class DarkCirclesMetric(BaseModel):
    severity: Optional[str] = None
    score: Optional[float] = None
    darkness_score: Optional[float] = None
    color_score: Optional[float] = None
    left_eye: Optional[float] = None
    right_eye: Optional[float] = None
    description: Optional[str] = None


class RednessMetric(BaseModel):
    severity: Optional[str] = None
    score: Optional[float] = None
    percentage: Optional[float] = None
    red_intensity: Optional[float] = None
    red_dominance: Optional[float] = None
    description: Optional[str] = None


class SkinMetrics(BaseModel):
    acne: Optional[AcneMetric] = None
    oiliness: Optional[OilinessMetric] = None
    pigmentation: Optional[PigmentationMetric] = None
    wrinkles: Optional[WrinklesMetric] = None
    pores: Optional[PoresMetric] = None
    hydration: Optional[HydrationMetric] = None
    darkCircles: Optional[DarkCirclesMetric] = Field(default=None, alias="darkCircles")
    redness: Optional[RednessMetric] = None

    class Config:
        populate_by_name = True


class SkinComprehensive(BaseModel):
    skinType: Optional[str] = None
    concerns: Optional[List[str]] = None
    skinHealth: Optional[int] = None
    recommendations: Optional[List[str]] = None
    priorityActions: Optional[List[str]] = None


class SkinAnalysisDoc(BaseModel):
    clerk_user_id: str
    created_at: datetime
    image: str
    metrics: SkinMetrics
    comprehensive: SkinComprehensive
    face_region: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
