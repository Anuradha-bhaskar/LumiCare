import logging
from typing import Any, Dict

import numpy as np

from .skin_analysis_methods import skin_analysis_methods
from .skin_comprehensive_analyzer import skin_comprehensive_analyzer
from .skin_image_processor import skin_image_processor
from .skin_metrics_analyzer import skin_metrics_analyzer

logger = logging.getLogger(__name__)

class SkinAnalyzer:
    """Main skin analyzer class that orchestrates all analysis components"""
    
    def __init__(self):
        self.image_processor = skin_image_processor
        self.metrics_analyzer = skin_metrics_analyzer
        self.analysis_methods = skin_analysis_methods
        self.comprehensive_analyzer = skin_comprehensive_analyzer

    def analyze_skin(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            # Process image and extract face region
            landmarks, face_region = self.image_processor.process_image_for_analysis(image)
            
            # Perform all individual metric analyses
            analysis = {
                "acne": self.metrics_analyzer.analyze_acne(face_region),
                "oiliness": self.analysis_methods.analyze_oiliness(face_region),
                "pigmentation": self.analysis_methods.analyze_pigmentation(face_region),
                "wrinkles": self.analysis_methods.analyze_wrinkles(face_region),
                "pores": self.analysis_methods.analyze_pores(face_region),
                "hydration": self.analysis_methods.analyze_hydration(face_region),
                "darkCircles": self.analysis_methods.analyze_dark_circles(image, landmarks),
                "redness": self.analysis_methods.analyze_redness(face_region)
            }
            
            # Generate comprehensive analysis using AI
            comprehensive_analysis = self.comprehensive_analyzer.generate_comprehensive_analysis(analysis)
            
            return {
                "metrics": analysis,
                "comprehensive": comprehensive_analysis,
                "face_region": self.image_processor.encode_face_region(face_region)
            }
        except Exception as e:
            logger.error(f"Error in skin analysis: {e}")
            raise

# Global instance
skin_analyzer = SkinAnalyzer()
