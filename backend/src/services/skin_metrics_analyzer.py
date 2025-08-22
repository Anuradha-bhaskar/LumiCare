import logging
from typing import Any, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class SkinMetricsAnalyzer:
    """Analyzes individual skin metrics like acne"""
    
    def analyze_acne(self, face_region: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # More restrictive HSV ranges for red/pink acne spots
            # Increased saturation threshold to avoid detecting normal skin tones
            lower_red1 = np.array([0, 70, 50])      # Increased saturation from 40 to 70
            upper_red1 = np.array([10, 255, 255])  # Narrowed hue range from 15 to 10
            lower_red2 = np.array([170, 70, 50])   # Increased saturation, narrowed hue range
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks for red regions
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Enhanced texture analysis with higher threshold
            gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
            laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)
            texture_mask = np.uint8(np.absolute(laplacian))
            
            # Apply higher threshold for texture mask to reduce false positives
            _, texture_mask = cv2.threshold(texture_mask, 50, 255, cv2.THRESH_BINARY)
            
            # Combine red mask with texture mask
            combined_mask = cv2.bitwise_and(red_mask, texture_mask)
            
            # More aggressive morphological operations to clean up noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Larger kernel
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Additional erosion to remove small noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.erode(combined_mask, kernel_small, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            acne_spots: List[Dict[str, int]] = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # More restrictive size filtering
                if 50 < area < 500:  # Increased minimum area from 30 to 50, reduced max from 800 to 500
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # More restrictive aspect ratio and additional shape validation
                    if 0.6 < aspect_ratio < 1.7:  # More circular shapes only
                        # Additional validation: check circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            # Only accept reasonably circular shapes
                            if circularity > 0.3:  # Circularity threshold
                                # Check if the spot has sufficient contrast with surrounding area
                                roi = gray[y:y+h, x:x+w]
                                if roi.size > 0:
                                    spot_mean = np.mean(roi)
                                    
                                    # Get surrounding area for contrast check
                                    margin = 5
                                    y_start = max(0, y - margin)
                                    y_end = min(gray.shape[0], y + h + margin)
                                    x_start = max(0, x - margin)
                                    x_end = min(gray.shape[1], x + w + margin)
                                    
                                    surrounding_roi = gray[y_start:y_end, x_start:x_end]
                                    surrounding_mean = np.mean(surrounding_roi)
                                    
                                    # Only consider it acne if there's sufficient contrast
                                    contrast_ratio = abs(spot_mean - surrounding_mean) / surrounding_mean if surrounding_mean > 0 else 0
                                    
                                    if contrast_ratio > 0.15:  # Minimum contrast threshold
                                        acne_spots.append({
                                            "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                                            "area": int(area)
                                        })
            
            num_spots = len(acne_spots)
            avg_size = np.mean([spot["area"] for spot in acne_spots]) if acne_spots else 0
            
            # More conservative severity assessment
            if num_spots == 0:
                severity = "Clear"
                score = 0
            elif num_spots <= 2:  # Reduced from 3 to 2
                severity = "Mild"
                score = 0.2
            elif num_spots <= 5:  # Reduced from 8 to 5
                severity = "Moderate"
                score = 0.4
            elif num_spots <= 10:  # Reduced from 15 to 10
                severity = "Moderate-Severe"
                score = 0.7
            else:
                severity = "Severe"
                score = 1.0
            
            return {
                "severity": severity,
                "count": num_spots,
                "score": score,
                "avg_size": float(avg_size),
                "spots": acne_spots,
                "description": f"{severity} acne ({num_spots} spots detected)"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing acne: {e}")
            # Return more specific error information for debugging
            return {
                "severity": "Error",
                "count": 0,
                "score": 0,
                "avg_size": 0,
                "spots": [],
                "description": f"Analysis failed: {str(e)}"
            }

# Global instance
skin_metrics_analyzer = SkinMetricsAnalyzer()
