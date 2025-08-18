from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import base64
from typing import Dict, Any, List
import logging
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional
from fastapi import Depends, Request
from ..database.mongo import get_db
from bson import ObjectId
from ..models.analysis import SkinAnalysisDoc, SkinMetrics, SkinComprehensive

load_dotenv()

router = APIRouter(prefix="/api/skin", tags=["skin"])

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinAnalysisRequest(BaseModel):
    image_data: str
    clerk_user_id: Optional[str] = None

class SkinAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def analyze_skin(self, image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive skin analysis using computer vision"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get face landmarks
            results = self.mp_face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                raise Exception("No face detected in image")
            
            landmarks = results.multi_face_landmarks[0]
            
            # Extract face region with better bounds
            face_region = self._extract_face_region(image, landmarks)
            
            # Analyze different skin metrics
            analysis = {
                "acne": self._analyze_acne(face_region),
                "oiliness": self._analyze_oiliness(face_region),
                "pigmentation": self._analyze_pigmentation(face_region),
                "wrinkles": self._analyze_wrinkles(face_region),
                "pores": self._analyze_pores(face_region),
                "hydration": self._analyze_hydration(face_region),
                "darkCircles": self._analyze_dark_circles(image, landmarks),
                "redness": self._analyze_redness(face_region)
            }
            
            # Generate comprehensive analysis with Gemini
            comprehensive_analysis = self._generate_comprehensive_analysis(analysis)
            
            return {
                "metrics": analysis,
                "comprehensive": comprehensive_analysis,
                "face_region": self._encode_face_region(face_region)
            }
            
        except Exception as e:
            logger.error(f"Error in skin analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Skin analysis failed: {str(e)}")
    
    def _extract_face_region(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Extract the face region for analysis with improved bounds"""
        height, width = image.shape[:2]
        
        # Get face contour landmarks (more reliable than all landmarks)
        face_oval_landmarks = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Get coordinates
        x_coords = []
        y_coords = []
        
        for landmark_id in face_oval_landmarks:
            if landmark_id < len(landmarks.landmark):
                landmark = landmarks.landmark[landmark_id]
                x_coords.append(landmark.x * width)
                y_coords.append(landmark.y * height)
        
        # If face oval landmarks fail, use all landmarks
        if not x_coords:
            x_coords = [landmark.x * width for landmark in landmarks.landmark]
            y_coords = [landmark.y * height for landmark in landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add reasonable padding
        padding_x = max(20, int((x_max - x_min) * 0.1))
        padding_y = max(20, int((y_max - y_min) * 0.1))
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(width, x_max + padding_x)
        y_max = min(height, y_max + padding_y)
        
        return image[y_min:y_max, x_min:x_max]
    
    def _analyze_acne(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Improved acne detection using multiple approaches"""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Method 1: Red color detection in HSV
            lower_red1 = np.array([0, 40, 40])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([160, 40, 40])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Method 2: Texture analysis for bumps
            gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
            laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)
            texture_mask = np.uint8(np.absolute(laplacian))
            
            # Combine masks
            combined_mask = cv2.bitwise_and(red_mask, texture_mask)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and aspect ratio
            acne_spots = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 30 < area < 800:  # Adjusted size range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Filter by aspect ratio (acne spots are roughly circular)
                    if 0.5 < aspect_ratio < 2.0:
                        acne_spots.append({
                            "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                            "area": int(area)
                        })
            
            # Determine severity based on count and average size
            num_spots = len(acne_spots)
            avg_size = np.mean([spot["area"] for spot in acne_spots]) if acne_spots else 0
            
            # Calculate severity score
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
            return {"severity": "Unknown", "count": 0, "score": 0, "avg_size": 0, "spots": [], "description": "Analysis failed"}
    
    def _analyze_oiliness(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Improved oiliness analysis using brightness and texture patterns"""
        try:
            # Convert to LAB color space for better analysis
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]  # Lightness channel
            
            # Define T-zone (forehead, nose, chin) - typically more oily
            height, width = face_region.shape[:2]
            
            # T-zone mask
            t_zone_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Forehead region
            cv2.rectangle(t_zone_mask, (width//4, 0), (3*width//4, height//3), 255, -1)
            
            # Nose region (center vertical strip)
            cv2.rectangle(t_zone_mask, (2*width//5, height//4), (3*width//5, 3*height//4), 255, -1)
            
            # Calculate brightness in T-zone vs other areas
            t_zone_brightness = np.mean(l_channel[t_zone_mask == 255])
            other_brightness = np.mean(l_channel[t_zone_mask == 0]) if np.any(t_zone_mask == 0) else t_zone_brightness
            
            # Calculate texture variance (oily skin often has different texture)
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            texture_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            
            # Normalize brightness difference (0-1)
            brightness_diff = min(abs(t_zone_brightness - other_brightness) / 50.0, 1.0)
            
            # Combine metrics
            oiliness_score = (brightness_diff * 0.6) + (min(texture_variance / 2000, 1.0) * 0.4)
            
            # Determine severity
            if oiliness_score < 0.2:
                severity = "Dry"
            elif oiliness_score < 0.4:
                severity = "Normal"
            elif oiliness_score < 0.7:
                severity = "Oily"
            else:
                severity = "Very Oily"
            
            return {
                "severity": severity,
                "score": float(oiliness_score),
                "t_zone_brightness": float(t_zone_brightness),
                "texture_variance": float(texture_variance),
                "description": f"{severity} skin (oiliness score: {oiliness_score:.2f})"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing oiliness: {e}")
            return {"severity": "Unknown", "score": 0.0, "description": "Analysis failed"}
    
    def _analyze_pigmentation(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Improved pigmentation analysis using color uniformity"""
        try:
            # Convert to LAB color space for better color analysis
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]  # Lightness
            a_channel = lab[:, :, 1]  # Green-Red
            b_channel = lab[:, :, 2]  # Blue-Yellow
            
            # Calculate color uniformity
            l_std = np.std(l_channel)
            a_std = np.std(a_channel)
            b_std = np.std(b_channel)
            
            # Combined color variation
            color_variation = (l_std + a_std + b_std) / 3
            
            # Normalize to 0-1 scale
            normalized_variation = min(color_variation / 30.0, 1.0)
            
            # Detect potential dark spots using thresholding
            mean_lightness = np.mean(l_channel)
            dark_threshold = mean_lightness - (l_std * 1.5)
            dark_spots_mask = l_channel < dark_threshold
            dark_spots_percentage = np.sum(dark_spots_mask) / dark_spots_mask.size
            
            # Combine metrics
            pigmentation_score = (normalized_variation * 0.7) + (dark_spots_percentage * 0.3)
            
            # Determine severity
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
    
    def _analyze_wrinkles(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Improved wrinkle detection using directional filters"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Use multiple edge detection methods
            # Canny edge detection
            canny = cv2.Canny(blurred, 30, 80)
            
            # Sobel filters for different directions
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            # Combine edge information
            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))
            
            # Focus on lines (potential wrinkles)
            kernel_horizontal = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
            kernel_vertical = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
            
            horizontal_lines = cv2.filter2D(blurred, -1, kernel_horizontal)
            vertical_lines = cv2.filter2D(blurred, -1, kernel_vertical)
            
            # Combine all edge information
            combined_edges = cv2.bitwise_or(canny, sobel_normalized)
            
            # Calculate wrinkle metrics
            total_pixels = combined_edges.shape[0] * combined_edges.shape[1]
            edge_pixels = np.sum(combined_edges > 0)
            wrinkle_density = edge_pixels / total_pixels
            
            # Analyze line continuity (wrinkles are continuous lines)
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            long_lines = [c for c in contours if cv2.arcLength(c, False) > 20]
            line_score = len(long_lines) / max(len(contours), 1)
            
            # Combine metrics
            wrinkle_score = (wrinkle_density * 100) * 0.6 + line_score * 0.4
            
            # Determine severity
            if wrinkle_score < 0.1:
                severity = "None"
            elif wrinkle_score < 0.3:
                severity = "Fine Lines"
            elif wrinkle_score < 0.6:
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
            return {"severity": "Unknown", "score": 0.0, "description": "Analysis failed"}
    
    def _analyze_pores(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Improved pore analysis using morphological operations"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply top-hat transform to detect dark spots
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            
            # Threshold to find dark spots
            _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up the binary image
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter pore-like contours
            pores = []
            total_pore_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Pores are typically small circular objects
                if 5 < area < 150:
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.3:  # Reasonably circular
                            pores.append(area)
                            total_pore_area += area
            
            # Calculate metrics
            pore_count = len(pores)
            avg_pore_size = np.mean(pores) if pores else 0
            pore_density = pore_count / (face_region.shape[0] * face_region.shape[1] / 10000)  # per 100x100 pixel area
            
            # Determine severity based on size and density
            size_score = min(avg_pore_size / 50.0, 1.0)
            density_score = min(pore_density / 5.0, 1.0)
            overall_score = (size_score + density_score) / 2
            
            if overall_score < 0.2:
                severity = "Fine"
            elif overall_score < 0.5:
                severity = "Normal"
            elif overall_score < 0.7:
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
    
    def _analyze_hydration(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Improved hydration analysis using texture and surface properties"""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Method 1: Texture smoothness analysis
            # Well-hydrated skin appears smoother
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            smoothness = max(0, 1 - (texture_variance / 1000))
            
            # Method 2: Local binary pattern for texture analysis
            # Create a simple LBP-like analysis
            kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            texture_response = cv2.filter2D(gray, -1, kernel)
            texture_uniformity = 1 - (np.std(texture_response) / 255.0)
            
            # Method 3: Surface reflection analysis
            # Hydrated skin has different reflection properties
            blur_small = cv2.GaussianBlur(gray, (5, 5), 0)
            blur_large = cv2.GaussianBlur(gray, (15, 15), 0)
            reflection_diff = np.mean(np.abs(blur_small.astype(float) - blur_large.astype(float)))
            reflection_score = min(reflection_diff / 20.0, 1.0)
            
            # Combine all metrics
            hydration_score = (smoothness * 0.4) + (texture_uniformity * 0.4) + (reflection_score * 0.2)
            
            # Determine severity
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
    
    def _analyze_dark_circles(self, image: np.ndarray, landmarks) -> Dict[str, Any]:
        """Improved dark circles analysis with better eye region detection"""
        try:
            height, width = image.shape[:2]
            
            # More accurate eye region landmarks
            left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Under-eye specific landmarks for better targeting
            left_under_eye = [159, 158, 157, 173, 133, 155, 154, 153]
            right_under_eye = [385, 386, 387, 388, 466, 263, 249, 390]
            
            # Get under-eye regions
            left_region = self._get_under_eye_region(image, landmarks, left_under_eye, width, height)
            right_region = self._get_under_eye_region(image, landmarks, right_under_eye, width, height)
            
            # Analyze darkness with improved method
            left_darkness = self._calculate_advanced_darkness(left_region)
            right_darkness = self._calculate_advanced_darkness(right_region)
            
            avg_darkness = (left_darkness + right_darkness) / 2
            
            # Color analysis for more accurate assessment
            left_color_score = self._analyze_under_eye_color(left_region)
            right_color_score = self._analyze_under_eye_color(right_region)
            avg_color_score = (left_color_score + right_color_score) / 2
            
            # Combine darkness and color metrics
            final_score = (avg_darkness * 0.6) + (avg_color_score * 0.4)
            
            # Determine severity
            if final_score < 0.2:
                severity = "None"
            elif final_score < 0.4:
                severity = "Mild"
            elif final_score < 0.6:
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
    
    def _analyze_redness(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Improved redness analysis using multiple color spaces"""
        try:
            # Convert to multiple color spaces for better red detection
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Method 1: HSV red detection with refined ranges
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 30, 30])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Method 2: LAB color space analysis
            # In LAB, 'a' channel represents green-red axis
            a_channel = lab[:, :, 1]
            # Higher 'a' values indicate more red
            red_intensity = np.mean(a_channel[a_channel > np.mean(a_channel)])
            
            # Method 3: BGR analysis
            b, g, r = cv2.split(face_region)
            # Calculate red dominance
            red_dominance = np.mean(r.astype(float) / (b.astype(float) + g.astype(float) + r.astype(float) + 1e-6))
            
            # Calculate redness percentage from mask
            red_pixels = np.sum(red_mask > 0)
            total_pixels = red_mask.shape[0] * red_mask.shape[1]
            redness_percentage = red_pixels / total_pixels
            
            # Combine all metrics
            normalized_red_intensity = min((red_intensity - 128) / 50.0, 1.0) if red_intensity > 128 else 0
            redness_score = (redness_percentage * 0.4) + (normalized_red_intensity * 0.3) + (red_dominance * 0.3)
            
            # Determine severity
            if redness_score < 0.1:
                severity = "None"
            elif redness_score < 0.25:
                severity = "Mild"
            elif redness_score < 0.5:
                severity = "Moderate"
            else:
                severity = "Severe"
            
            return {
                "severity": severity,
                "score": float(redness_score),
                "percentage": float(redness_percentage),
                "red_intensity": float(red_intensity),
                "red_dominance": float(red_dominance),
                "description": f"{severity} redness (score: {redness_score:.2f})"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing redness: {e}")
            return {"severity": "Unknown", "score": 0.0, "percentage": 0.0, "description": "Analysis failed"}
    
    def _get_under_eye_region(self, image: np.ndarray, landmarks, eye_landmarks: List[int], width: int, height: int) -> np.ndarray:
        """Extract under-eye region more accurately"""
        try:
            # Get eye landmark coordinates
            eye_coords = []
            for landmark_id in eye_landmarks:
                if landmark_id < len(landmarks.landmark):
                    landmark = landmarks.landmark[landmark_id]
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    eye_coords.append((x, y))
            
            if len(eye_coords) < 3:
                return np.array([])
            
            # Get bounding box with focus on under-eye area
            x_coords = [coord[0] for coord in eye_coords]
            y_coords = [coord[1] for coord in eye_coords]
            
            x_min, x_max = max(0, min(x_coords)), min(width, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(height, max(y_coords))
            
            # Extend the region downward to capture under-eye area
            y_extension = int((y_max - y_min) * 0.8)
            y_max = min(height, y_max + y_extension)
            
            # Add horizontal padding
            padding = max(5, int((x_max - x_min) * 0.1))
            x_min = max(0, x_min - padding)
            x_max = min(width, x_max + padding)
            
            return image[y_min:y_max, x_min:x_max]
            
        except Exception as e:
            logger.error(f"Error extracting under-eye region: {e}")
            return np.array([])
    
    def _calculate_advanced_darkness(self, region: np.ndarray) -> float:
        """Advanced darkness calculation with multiple metrics"""
        if region.size == 0:
            return 0.0
        
        try:
            # Convert to LAB for better lightness analysis
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Calculate relative darkness
            mean_lightness = np.mean(l_channel)
            darkness_score = 1 - (mean_lightness / 255.0)
            
            # Also consider the lower percentile (darkest areas)
            bottom_10_percentile = np.percentile(l_channel, 10)
            extreme_darkness = 1 - (bottom_10_percentile / 255.0)
            
            # Combine scores
            final_darkness = (darkness_score * 0.7) + (extreme_darkness * 0.3)
            
            return min(final_darkness, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating darkness: {e}")
            return 0.0
    
    def _analyze_under_eye_color(self, region: np.ndarray) -> float:
        """Analyze color characteristics of under-eye region"""
        if region.size == 0:
            return 0.0
        
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Dark circles often have purple/blue tints
            # Check for these color ranges
            lower_purple = np.array([120, 30, 30])
            upper_purple = np.array([150, 255, 255])
            
            lower_blue = np.array([100, 30, 30])
            upper_blue = np.array([130, 255, 255])
            
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            color_mask = cv2.bitwise_or(purple_mask, blue_mask)
            
            # Calculate color score
            color_pixels = np.sum(color_mask > 0)
            total_pixels = color_mask.shape[0] * color_mask.shape[1]
            
            if total_pixels > 0:
                color_score = color_pixels / total_pixels
            else:
                color_score = 0.0
            
            return min(color_score * 2, 1.0)  # Amplify the score slightly
            
        except Exception as e:
            logger.error(f"Error analyzing under-eye color: {e}")
            return 0.0
    
    def _encode_face_region(self, face_region: np.ndarray) -> str:
        """Encode face region as base64 for frontend display"""
        try:
            if face_region.size == 0:
                return ""
            
            _, buffer = cv2.imencode('.jpg', face_region, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Error encoding face region: {e}")
            return ""
    
    def _generate_comprehensive_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis using Gemini"""
        try:
            # Create a more structured prompt for Gemini
            analysis_summary = {}
            
            # Extract key metrics
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
                "skinType": "one of: Oily, Dry, Combination, Normal, Sensitive",
                "concerns": ["list of top 3 concerns"],
                "skinHealth": number from 0-100,
                "recommendations": ["4-5 short specific actionable recommendations"],
                "priorityActions": ["immediate actions to take"]
            }}

            Base your analysis on the severity levels provided. Focus on practical, actionable advice.
            """
            
            response = model.generate_content(prompt)
            
            # Try to parse JSON response
            try:
                import json
                # Clean the response text
                response_text = response.text.strip()
                
                # Find JSON content between braces
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx]
                    analysis = json.loads(json_text)
                    
                    # Validate required fields
                    required_fields = ['skinType', 'concerns', 'skinHealth', 'recommendations', 'priorityActions']
                    if all(field in analysis for field in required_fields):
                        return analysis
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse Gemini JSON response: {e}")
            
            # Fallback analysis based on metrics
            return self._generate_fallback_analysis(metrics)
                
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            return self._generate_fallback_analysis(metrics)
    
    def _generate_fallback_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback analysis when Gemini fails"""
        try:
            concerns = []
            recommendations = []
            priority_actions = []
            
            # Analyze each metric
            oiliness = metrics.get('oiliness', {}).get('severity', 'Normal')
            acne = metrics.get('acne', {}).get('severity', 'Clear')
            hydration = metrics.get('hydration', {}).get('severity', 'Adequately Hydrated')
            pigmentation = metrics.get('pigmentation', {}).get('severity', 'Even')
            
            # Determine skin type
            if 'Oily' in oiliness or 'Very Oily' in oiliness:
                if 'Dehydrated' in hydration:
                    skin_type = "Oily-Dehydrated"
                else:
                    skin_type = "Oily"
            elif 'Dry' in oiliness or 'Dehydrated' in hydration:
                skin_type = "Dry"
            elif metrics.get('redness', {}).get('severity') in ['Moderate', 'Severe']:
                skin_type = "Sensitive"
            else:
                skin_type = "Combination"
            
            # Identify concerns
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
            
            # Calculate skin health score
            severity_scores = {
                'None': 100, 'Clear': 100, 'Fine': 90, 'Even': 95,
                'Mild': 80, 'Normal': 85, 'Adequately Hydrated': 85,
                'Moderate': 60, 'Enlarged': 60,
                'Severe': 30, 'Very Oily': 50, 'Severely Dehydrated': 40
            }
            
            scores = []
            for metric_data in metrics.values():
                if isinstance(metric_data, dict) and 'severity' in metric_data:
                    severity = metric_data['severity']
                    scores.append(severity_scores.get(severity, 70))
            
            skin_health = int(np.mean(scores)) if scores else 75
            
            # Ensure we have at least some concerns and recommendations
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
                "concerns": concerns[:3],  # Top 3 concerns
                "skinHealth": skin_health,
                "recommendations": recommendations[:5],  # Top 5 recommendations
                "priorityActions": priority_actions[:3]  # Top 3 priority actions
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

# Global analyzer instance
analyzer = SkinAnalyzer()

@router.post("/analyze")
async def analyze_skin(request: SkinAnalysisRequest, db=Depends(get_db)) -> JSONResponse:
    """Analyze skin using improved computer vision and AI"""
    try:
        # Decode base64 image
        image_data = request.image_data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Validate image size and quality
        height, width = image.shape[:2]
        if width < 200 or height < 200:
            raise HTTPException(status_code=400, detail="Image too small. Please use an image at least 200x200 pixels")
        
        # Perform skin analysis
        results = analyzer.analyze_skin(image)
        
        logger.info(f"Skin analysis completed successfully for image {width}x{height}")
        
        # Persist to DB if user id provided
        saved_to_db = False
        try:
            if request.clerk_user_id:
                logger.info(f"Attempting to save analysis for user {request.clerk_user_id}")
                metrics_model = SkinMetrics.model_validate(results.get("metrics", {}))
                comprehensive_model = SkinComprehensive.model_validate(results.get("comprehensive", {}))
                doc_model = SkinAnalysisDoc(
                    clerk_user_id=request.clerk_user_id,
                    created_at=datetime.utcnow(),
                    image=request.image_data,
                    metrics=metrics_model,
                    comprehensive=comprehensive_model,
                    face_region=results.get("face_region", "")
                )
                insert_res = await db.skin_analyses.insert_one(doc_model.model_dump(by_alias=True))
                saved_to_db = insert_res.inserted_id is not None
                logger.info(f"Skin analysis saved: {saved_to_db}, id={insert_res.inserted_id}")
            else:
                logger.info("No clerk_user_id provided; skipping DB save")
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
    """Fetch recent skin analysis history for a user"""
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
