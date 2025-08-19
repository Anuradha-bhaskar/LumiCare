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
            
            # More specific color ranges for inflamed acne (avoiding deep reds of bindis/moles)
            # Focus on pinkish-red inflammation rather than deep reds
            lower_red1 = np.array([0, 30, 80])    # Lighter, more inflamed looking reds
            upper_red1 = np.array([10, 180, 255])
            lower_red2 = np.array([170, 30, 80])
            upper_red2 = np.array([180, 180, 255])
            
            # Additional pink range for lighter acne
            lower_pink = np.array([160, 20, 100])
            upper_pink = np.array([180, 100, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask3 = cv2.inRange(hsv, lower_pink, upper_pink)
            
            color_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
            
            # Enhanced texture analysis to detect raised/irregular surfaces
            gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Multiple edge detection methods
            laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)
            sobel_x = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gaussian, cv2.CV_64F, 0, 1, ksize=3)
            
            # Combine edge responses
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            texture_mask = np.uint8(np.absolute(laplacian) + edges/2)
            
            # Threshold for texture (acne has more irregular texture than smooth bindis/moles)
            _, texture_binary = cv2.threshold(texture_mask, 20, 255, cv2.THRESH_BINARY)
            
            # Combine color and texture information
            combined_mask = cv2.bitwise_and(color_mask, texture_binary)
            
            # Morphological operations to clean up
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Remove noise
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
            # Fill small gaps
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            acne_spots: List[Dict[str, int]] = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Refined size filtering (acne is typically smaller than bindis/large moles)
                if 15 < area < 400:  # Smaller upper limit to exclude large decorative items
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate additional features
                    aspect_ratio = w / h if h > 0 else 0
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # More strict filtering criteria
                    if (0.4 < aspect_ratio < 2.5 and  # Allow slightly more elongated shapes for acne
                        circularity < 0.85):  # Exclude very circular objects (likely bindis/moles)
                        
                        # Additional check: analyze the region's color uniformity
                        mask_region = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.fillPoly(mask_region, [contour], 255)
                        region_pixels = face_region[mask_region == 255]
                        
                        if len(region_pixels) > 0:
                            # Check color variance (acne typically has more color variation)
                            color_std = np.std(region_pixels, axis=0).mean()
                            
                            # Check if it's positioned in typical acne locations (avoid forehead center for bindis)
                            face_height, face_width = face_region.shape[:2]
                            center_x, center_y = x + w//2, y + h//2
                            
                            # Avoid the upper-center forehead area where bindis are typically placed
                            is_bindi_region = (center_y < face_height * 0.4 and 
                                            abs(center_x - face_width/2) < face_width * 0.15)
                            
                            # Filter out based on color uniformity and position
                            if color_std > 8 and not is_bindi_region:  # Acne has more color variation
                                acne_spots.append({
                                    "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                                    "area": int(area)
                                })
            
            # Calculate results
            num_spots = len(acne_spots)
            avg_size = np.mean([spot["area"] for spot in acne_spots]) if acne_spots else 0
            
            # Severity classification
            if num_spots == 0:
                severity = "Clear"
                score = 0
            elif num_spots <= 3:
                severity = "Mild"
                score = 0.25
            elif num_spots <= 8:
                severity = "Moderate"
                score = 0.5
            elif num_spots <= 15:
                severity = "Moderate-Severe"
                score = 0.75
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
            return {
                "severity": "Unknown", 
                "count": 0, 
                "score": 0, 
                "avg_size": 0, 
                "spots": [], 
                "description": "Analysis failed"
            }

# Global instance
skin_metrics_analyzer = SkinMetricsAnalyzer()
