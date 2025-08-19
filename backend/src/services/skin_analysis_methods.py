import logging
from typing import Any, Dict, List

import cv2
import numpy as np
from .skin_image_processor import skin_image_processor

logger = logging.getLogger(__name__)

class SkinAnalysisMethods:
    """Contains all individual skin analysis methods"""
    
    def analyze_oiliness(self, face_region: np.ndarray) -> Dict[str, Any]:
        try:
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            l_channel = lab[:, :, 0]
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]
            
            height, width = face_region.shape[:2]
            t_zone_mask = np.zeros((height, width), dtype=np.uint8)
            
            cv2.rectangle(t_zone_mask, (width//5, height//10), (4*width//5, height//3), 255, -1)
            cv2.rectangle(t_zone_mask, (2*width//5, height//4), (3*width//5, 2*height//3), 255, -1)
            cv2.ellipse(t_zone_mask, (width//2, 4*height//5), (width//6, height//8), 0, 0, 360, 255, -1)
            
            t_zone_brightness = np.mean(v_channel[t_zone_mask == 255])
            other_brightness = np.mean(v_channel[t_zone_mask == 0]) if np.any(t_zone_mask == 0) else t_zone_brightness
            
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            gray_std = np.std(gray)
            
            brightness_diff = min(abs(t_zone_brightness - other_brightness) / 30.0, 1.0)
            smoothness_score = 1.0 - min(gray_std / 50.0, 1.0)
            texture_score = min(laplacian_var / 1000.0, 1.0)
            
            oiliness_score = (brightness_diff * 0.3 + smoothness_score * 0.4 + (1.0 - texture_score) * 0.3)
            oiliness_score = min(oiliness_score + 0.15, 1.0)
            
            if oiliness_score < 0.25:
                severity = "Dry"
            elif oiliness_score < 0.45:
                severity = "Normal"
            elif oiliness_score < 0.70:
                severity = "Oily"
            else:
                severity = "Very Oily"
            
            return {
                "severity": severity,
                "score": float(oiliness_score),
                "t_zone_brightness": float(t_zone_brightness),
                "texture_variance": float(laplacian_var),
                "description": f"{severity} skin (oiliness score: {oiliness_score:.2f})"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing oiliness: {e}")
            return {"severity": "Unknown", "score": 0.0, "description": "Analysis failed"}

    def analyze_pigmentation(self, face_region: np.ndarray) -> Dict[str, Any]:
        try:
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            a_channel = lab[:, :, 1]
            b_channel = lab[:, :, 2]
            l_std = np.std(l_channel)
            a_std = np.std(a_channel)
            b_std = np.std(b_channel)
            color_variation = (l_std + a_std + b_std) / 3
            normalized_variation = min(color_variation / 30.0, 1.0)
            mean_lightness = np.mean(l_channel)
            dark_threshold = mean_lightness - (l_std * 1.5)
            dark_spots_mask = l_channel < dark_threshold
            dark_spots_percentage = np.sum(dark_spots_mask) / dark_spots_mask.size
            pigmentation_score = (normalized_variation * 0.7) + (dark_spots_percentage * 0.3)
            if pigmentation_score < 0.2:
                severity = "Even"
            elif pigmentation_score < 0.4:
                severity = "Mild Variation"
            elif pigmentation_score < 0.6:
                severity = "Moderate Variation"
            else:
                severity = "Significant Variation"
            return {
                "severity": severity,
                "score": float(pigmentation_score),
                "color_variation": float(color_variation),
                "dark_spots_percentage": float(dark_spots_percentage),
                "description": f"{severity} in skin tone"
            }
        except Exception as e:
            logger.error(f"Error analyzing pigmentation: {e}")
            return {"severity": "Unknown", "score": 0.0, "description": "Analysis failed"}

    def analyze_wrinkles(self, face_region: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
            enhanced = cv2.add(blurred, tophat)
            enhanced = cv2.subtract(enhanced, blackhat)
            
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            
            kernel_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
            kernel_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
            
            resp_h = cv2.filter2D(enhanced, cv2.CV_32F, kernel_h)
            resp_v = cv2.filter2D(enhanced, cv2.CV_32F, kernel_v)
            
            combined_response = np.maximum(np.abs(resp_h), np.abs(resp_v))
            combined_response = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX)
            combined_response = combined_response.astype(np.uint8)
            
            wrinkle_mask = cv2.adaptiveThreshold(combined_response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            wrinkle_mask = cv2.morphologyEx(wrinkle_mask, cv2.MORPH_OPEN, kernel_clean)
            
            contours, _ = cv2.findContours(wrinkle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            wrinkle_contours = []
            for contour in contours:
                perimeter = cv2.arcLength(contour, False)
                if perimeter < 30:
                    continue
                    
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width == 0 or height == 0:
                    continue
                    
                aspect_ratio = max(width, height) / min(width, height)
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if aspect_ratio > 5.0 and solidity > 0.5 and perimeter > 40:
                    wrinkle_contours.append(contour)
            
            total_pixels = wrinkle_mask.shape[0] * wrinkle_mask.shape[1]
            edge_pixels = np.sum(wrinkle_mask > 0)
            wrinkle_density = edge_pixels / total_pixels
            
            long_lines = [c for c in wrinkle_contours if cv2.arcLength(c, False) > 50]
            line_score = len(long_lines) / max(len(wrinkle_contours), 1) if wrinkle_contours else 0
            
            wrinkle_score = (wrinkle_density * 100) * 0.7 + line_score * 0.3
            
            if wrinkle_score < 0.02:
                severity = "None"
            elif wrinkle_score < 0.08:
                severity = "Fine Lines"
            elif wrinkle_score < 0.25:
                severity = "Moderate"
            else:
                severity = "Pronounced"
            
            return {
                "severity": severity,
                "score": float(wrinkle_score),
                "density": float(wrinkle_density),
                "line_count": len(long_lines),
                "description": f"{severity} wrinkles detected"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing wrinkles: {e}")
            return {"severity": "Error", "score": 0.0, "density": 0.0, "line_count": 0, "description": "Error in wrinkle analysis"}

    def analyze_pores(self, face_region: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            
            _, binary = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
            
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
            
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            pores: List[float] = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 15 < area < 100:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.5:
                            pores.append(area)
            
            pore_count = len(pores)
            avg_pore_size = np.mean(pores) if pores else 0
            pore_density = pore_count / (face_region.shape[0] * face_region.shape[1] / 10000)
            
            size_score = min(avg_pore_size / 80.0, 1.0)
            density_score = min(pore_density / 8.0, 1.0)
            overall_score = (size_score + density_score) / 2
            
            if overall_score < 0.3:
                severity = "Fine"
            elif overall_score < 0.6:
                severity = "Normal"
            elif overall_score < 0.8:
                severity = "Enlarged"
            else:
                severity = "Very Enlarged"
            
            return {
                "severity": severity,
                "count": pore_count,
                "avg_size": float(avg_pore_size),
                "density": float(pore_density),
                "score": float(overall_score),
                "description": f"{severity} pores (avg size: {avg_pore_size:.1f})"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pores: {e}")
            return {"severity": "Unknown", "count": 0, "avg_size": 0.0, "density": 0.0, "score": 0.0, "description": "Analysis failed"}

    def analyze_hydration(self, face_region: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            smoothness = max(0, 1 - (texture_variance / 1000))
            kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            texture_response = cv2.filter2D(gray, -1, kernel)
            texture_uniformity = 1 - (np.std(texture_response) / 255.0)
            blur_small = cv2.GaussianBlur(gray, (5, 5), 0)
            blur_large = cv2.GaussianBlur(gray, (15, 15), 0)
            reflection_diff = np.mean(np.abs(blur_small.astype(float) - blur_large.astype(float)))
            reflection_score = min(reflection_diff / 20.0, 1.0)
            hydration_score = (smoothness * 0.4) + (texture_uniformity * 0.4) + (reflection_score * 0.2)
            if hydration_score > 0.75:
                severity = "Well Hydrated"
            elif hydration_score > 0.5:
                severity = "Adequately Hydrated"
            elif hydration_score > 0.25:
                severity = "Dehydrated"
            else:
                severity = "Severely Dehydrated"
            return {
                "severity": severity,
                "score": float(hydration_score),
                "smoothness": float(smoothness),
                "texture_uniformity": float(texture_uniformity),
                "description": f"{severity} (hydration score: {hydration_score:.2f})"
            }
        except Exception as e:
            logger.error(f"Error analyzing hydration: {e}")
            return {"severity": "Unknown", "score": 0.0, "description": "Analysis failed"}

    def analyze_redness(self, face_region: np.ndarray) -> Dict[str, Any]:
        try:
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            lower_red1 = np.array([0, 70, 70])
            upper_red1 = np.array([8, 255, 255])
            lower_red2 = np.array([172, 70, 70])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            a_channel = lab[:, :, 1]
            red_threshold = 145
            red_pixels_lab = a_channel > red_threshold
            
            red_intensity = np.mean(a_channel[red_pixels_lab]) if np.sum(red_pixels_lab) > 0 else 128
            
            b, g, r = cv2.split(face_region)
            brightness = (b.astype(float) + g.astype(float) + r.astype(float)) / 3
            bright_mask = brightness > 80
            
            if np.sum(bright_mask) > 0:
                r_bright = r[bright_mask].astype(float)
                g_bright = g[bright_mask].astype(float)
                b_bright = b[bright_mask].astype(float)
                
                total_bright = r_bright + g_bright + b_bright + 1e-6
                red_dominance = np.mean(r_bright / total_bright)
                red_advantage = np.mean((r_bright - g_bright) + (r_bright - b_bright)) / 255.0
                red_advantage = max(0, red_advantage)
            else:
                red_dominance = 0.33
                red_advantage = 0
            
            red_pixels = np.sum(red_mask > 0)
            total_pixels = red_mask.shape[0] * red_mask.shape[1]
            redness_percentage = red_pixels / total_pixels
            
            normalized_red_intensity = max(0, (red_intensity - 150) / 35.0)
            normalized_red_intensity = min(normalized_red_intensity, 1.0)
            
            redness_score = (redness_percentage * 0.4 + normalized_red_intensity * 0.3 + red_dominance * 0.2 + red_advantage * 0.1)
            
            if redness_score < 0.08:
                redness_score = 0.0
            
            if redness_score < 0.12:
                severity = "None"
            elif redness_score < 0.25:
                severity = "Mild"
            elif redness_score < 0.45:
                severity = "Moderate"
            else:
                severity = "Severe"
            
            return {
                "severity": severity,
                "score": float(redness_score),
                "percentage": float(redness_percentage),
                "red_intensity": float(red_intensity),
                "red_dominance": float(red_dominance),
                "red_advantage": float(red_advantage),
                "description": f"{severity} redness (score: {redness_score:.3f})"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing redness: {e}")
            return {"severity": "Unknown", "score": 0.0, "percentage": 0.0, "description": "Analysis failed"}

    def analyze_dark_circles(self, image: np.ndarray, landmarks) -> Dict[str, Any]:
        try:
            height, width = image.shape[:2]
            left_under_eye = [159, 158, 157, 173, 133, 155, 154, 153]
            right_under_eye = [385, 386, 387, 388, 466, 263, 249, 390]
            left_region = skin_image_processor.get_under_eye_region(image, landmarks, left_under_eye, width, height)
            right_region = skin_image_processor.get_under_eye_region(image, landmarks, right_under_eye, width, height)
            left_darkness = self._calculate_advanced_darkness(left_region)
            right_darkness = self._calculate_advanced_darkness(right_region)
            avg_darkness = (left_darkness + right_darkness) / 2
            left_color_score = self._analyze_under_eye_color(left_region)
            right_color_score = self._analyze_under_eye_color(right_region)
            avg_color_score = (left_color_score + right_color_score) / 2
            final_score = (avg_darkness * 0.6) + (avg_color_score * 0.4)
            
            if final_score < 0.3:
                severity = "None"
            elif final_score < 0.5:
                severity = "Mild" 
            elif final_score < 0.7:
                severity = "Moderate"
            else:
                severity = "Severe"
            
            return {
                "severity": severity,
                "score": float(final_score),
                "darkness_score": float(avg_darkness),
                "color_score": float(avg_color_score),
                "left_eye": float(left_darkness),
                "right_eye": float(right_darkness),
                "description": f"{severity} dark circles (score: {final_score:.2f})"
            }
        except Exception as e:
            logger.error(f"Error analyzing dark circles: {e}")
            return {"severity": "Unknown", "score": 0.0, "description": "Analysis failed"}

    def _calculate_advanced_darkness(self, region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        try:
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            mean_lightness = np.mean(l_channel)
            darkness_score = 1 - (mean_lightness / 255.0)
            
            bottom_percentile = np.percentile(l_channel, 25)
            extreme_darkness = 1 - (bottom_percentile / 255.0)
            
            final_darkness = (darkness_score * 0.8) + (extreme_darkness * 0.2)
            baseline_adjusted = max(0, final_darkness - 0.15)
            
            return min(baseline_adjusted, 1.0)
        except Exception as e:
            logger.error(f"Error calculating darkness: {e}")
            return 0.0

    def _analyze_under_eye_color(self, region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            lower_purple = np.array([130, 50, 40])
            upper_purple = np.array([150, 255, 200])
            lower_blue = np.array([105, 50, 40])
            upper_blue = np.array([125, 255, 200])
            
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            color_mask = cv2.bitwise_or(purple_mask, blue_mask)
            
            color_pixels = np.sum(color_mask > 0)
            total_pixels = color_mask.shape[0] * color_mask.shape[1]
            color_score = (color_pixels / total_pixels) if total_pixels > 0 else 0.0
            
            return min(color_score * 1.5, 1.0)
        except Exception as e:
            logger.error(f"Error analyzing under-eye color: {e}")
            return 0.0

# Global instance
skin_analysis_methods = SkinAnalysisMethods()
