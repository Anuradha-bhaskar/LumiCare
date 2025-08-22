import json
import logging
import os
from typing import Any, Dict, List

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

class SkinComprehensiveAnalyzer:
    """Handles AI-powered comprehensive analysis and recommendations"""
    
    def generate_comprehensive_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        try:
            analysis_summary: Dict[str, Any] = {}
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'severity' in metric_data:
                    analysis_summary[metric_name] = {
                        'severity': metric_data['severity'],
                        'score': metric_data.get('score', 0)
                    }
            prompt = f"""
            Analyze the following skin assessment results and provide comprehensive recommendations:

            Skin Metrics:
            - Acne: {metrics.get('acne', {}).get('severity', 'Unknown')} ({metrics.get('acne', {}).get('count', 0)} spots)
            - Oiliness: {metrics.get('oiliness', {}).get('severity', 'Unknown')} 
            - Pigmentation: {metrics.get('pigmentation', {}).get('severity', 'Unknown')}
            - Wrinkles: {metrics.get('wrinkles', {}).get('severity', 'Unknown')}
            - Pores: {metrics.get('pores', {}).get('severity', 'Unknown')}
            - Hydration: {metrics.get('hydration', {}).get('severity', 'Unknown')}
            - Dark Circles: {metrics.get('darkCircles', {}).get('severity', 'Unknown')}
            - Redness: {metrics.get('redness', {}).get('severity', 'Unknown')}

            Please provide a JSON response with exactly these fields:
            {{
                "skinType": "one of: Oily, Dry, Combination, Normal",
                "concerns": ["list of up to 3 top concerns based on highest severity. If no major concerns, return ['No major concerns']"],
                "skinHealth": number from 0-100,
                "recommendations": ["4-5 short specific actionable recommendations"],
                "priorityActions": ["immediate actions to take"]
            }}

            Base your analysis on the severity levels provided. Focus on practical, actionable advice.
            """
            response = model.generate_content(prompt)
            try:
                response_text = (response.text or "").strip()
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx]
                    analysis = json.loads(json_text)
                    required_fields = ['skinType', 'concerns', 'skinHealth', 'recommendations', 'priorityActions']
                    if all(field in analysis for field in required_fields):
                        return analysis
            except Exception as e:
                logger.warning(f"Failed to parse Gemini JSON response: {e}")
            return self._generate_fallback_analysis(metrics)
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            return self._generate_fallback_analysis(metrics)

    def _generate_fallback_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        try:
            concerns: List[str] = []
            recommendations: List[str] = []
            priority_actions: List[str] = []
            oiliness = metrics.get('oiliness', {}).get('severity', 'Normal')
            acne = metrics.get('acne', {}).get('severity', 'Clear')
            hydration = metrics.get('hydration', {}).get('severity', 'Adequately Hydrated')
            pigmentation = metrics.get('pigmentation', {}).get('severity', 'Even')
            if 'Oily' in oiliness or 'Very Oily' in oiliness:
                skin_type = "Oily-Dehydrated" if 'Dehydrated' in hydration else "Oily"
            elif 'Dry' in oiliness or 'Dehydrated' in hydration:
                skin_type = "Dry"
            elif metrics.get('redness', {}).get('severity') in ['Moderate', 'Severe']:
                skin_type = "Sensitive"
            else:
                skin_type = "Combination"
            if acne not in ['Clear', 'None']:
                concerns.append("Acne")
                recommendations.extend([
                    "Use a gentle, non-comedogenic cleanser twice daily",
                    "Consider salicylic acid or benzoyl peroxide treatments",
                    "Avoid touching or picking at acne spots"
                ])
                priority_actions.append("Address acne with targeted treatments")
            if 'Dehydrated' in hydration or 'Severely Dehydrated' in hydration:
                concerns.append("Dehydration")
                recommendations.extend([
                    "Use a hydrating serum with hyaluronic acid",
                    "Apply a moisturizer suitable for your skin type",
                    "Increase water intake"
                ])
                priority_actions.append("Improve skin hydration")
            if pigmentation not in ['Even', 'None']:
                concerns.append("Pigmentation")
                recommendations.extend([
                    "Use sunscreen daily with SPF 30 or higher",
                    "Consider vitamin C serum for brightening",
                    "Use gentle exfoliation 1-2 times per week"
                ])
                priority_actions.append("Protect from sun damage")
            if metrics.get('darkCircles', {}).get('severity') in ['Moderate', 'Severe']:
                concerns.append("Dark Circles")
                recommendations.append("Use an eye cream with caffeine or vitamin C")
            severity_scores = {
                'None': 100, 'Clear': 100, 'Fine': 90, 'Even': 95,
                'Mild': 80, 'Normal': 85, 'Adequately Hydrated': 85,
                'Moderate': 60, 'Enlarged': 60,
                'Severe': 30, 'Very Oily': 50, 'Severely Dehydrated': 40
            }
            scores: List[int] = []
            for metric_data in metrics.values():
                if isinstance(metric_data, dict) and 'severity' in metric_data:
                    severity = metric_data['severity']
                    scores.append(severity_scores.get(severity, 70))
            skin_health = int(np.mean(scores)) if scores else 75
            if not concerns:
                concerns = ["Maintenance"]
            if not recommendations:
                recommendations = [
                    "Maintain a consistent daily skincare routine",
                    "Use sunscreen daily to prevent damage",
                    "Keep skin clean and moisturized"
                ]
            if not priority_actions:
                priority_actions = ["Establish a consistent skincare routine"]
            return {
                "skinType": skin_type,
                "concerns": concerns[:3],
                "skinHealth": skin_health,
                "recommendations": recommendations[:5],
                "priorityActions": priority_actions[:3]
            }
        except Exception as e:
            logger.error(f"Error in fallback analysis: {e}")
            return {
                "skinType": "Normal",
                "concerns": ["Assessment needed"],
                "skinHealth": 75,
                "recommendations": ["Consult with a dermatologist for proper assessment"],
                "priorityActions": ["Get professional skin evaluation"]
            }

# Global instance
skin_comprehensive_analyzer = SkinComprehensiveAnalyzer()
