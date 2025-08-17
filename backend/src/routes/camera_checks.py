from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import base64
from typing import Dict, Any
import logging
import json

router = APIRouter(prefix="/api/camera", tags=["camera"])

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class CameraAnalysisRequest(BaseModel):
    image_data: str

class CameraAnalyzer:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.previous_landmarks = None
        self.movement_threshold = 0.02  # Threshold for detecting movement
        
    def analyze_lighting(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting conditions in the image"""
        try:
            # Convert to grayscale for lighting analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness metrics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Calculate histogram to detect over/under exposure
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            
            # Check for over-exposure (too many bright pixels)
            over_exposed = bool(np.sum(hist_norm[200:]) > 0.1)
            
            # Check for under-exposure (too many dark pixels)
            under_exposed = bool(np.sum(hist_norm[:50]) > 0.3)
            
            # Calculate contrast
            contrast = std_brightness / (mean_brightness + 1e-6)
            
            # Determine lighting quality
            if over_exposed:
                lighting_quality = "over_exposed"
                is_good = False
            elif under_exposed:
                lighting_quality = "under_exposed"
                is_good = False
            elif mean_brightness < 80:
                lighting_quality = "too_dark"
                is_good = False
            elif mean_brightness > 200:
                lighting_quality = "too_bright"
                is_good = False
            elif contrast < 0.3:
                lighting_quality = "low_contrast"
                is_good = False
            else:
                lighting_quality = "good"
                is_good = True
            
            return {
                "is_good": bool(is_good),
                "quality": lighting_quality,
                "mean_brightness": float(mean_brightness),
                "contrast": float(contrast),
                "over_exposed": bool(over_exposed),
                "under_exposed": bool(under_exposed)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing lighting: {e}")
            return {"is_good": False, "quality": "error", "error": str(e)}
    
    def analyze_position(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face position and alignment"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {
                    "is_good": False,
                    "quality": "no_face_detected",
                    "message": "No face detected in the image"
                }
            
            # Get the first face landmarks
            landmarks = results.multi_face_landmarks[0]
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Key landmarks for position analysis
            # Nose tip (landmark 4)
            nose_tip = landmarks.landmark[4]
            nose_x, nose_y = nose_tip.x * width, nose_tip.y * height
            
            # Left eye (landmark 33)
            left_eye = landmarks.landmark[33]
            left_eye_x, left_eye_y = left_eye.x * width, left_eye.y * height
            
            # Right eye (landmark 263)
            right_eye = landmarks.landmark[263]
            right_eye_x, right_eye_y = right_eye.x * width, right_eye.y * height
            
            # Calculate face center
            face_center_x = (left_eye_x + right_eye_x) / 2
            face_center_y = (left_eye_y + right_eye_y) / 2
            
            # Check if face is centered horizontally
            center_tolerance = width * 0.15  # 15% tolerance
            is_centered_horizontal = bool(abs(face_center_x - width/2) < center_tolerance)
            
            # Check if face is centered vertically
            vertical_tolerance = height * 0.2  # 20% tolerance
            is_centered_vertical = bool(abs(face_center_y - height/2) < vertical_tolerance)
            
            # Check face size (should be reasonably large)
            face_width = abs(right_eye_x - left_eye_x) * 3  # Approximate face width
            min_face_width = width * 0.3  # Face should be at least 30% of image width
            is_face_size_good = bool(face_width > min_face_width)
            
            # Check if face is too close or too far
            max_face_width = width * 0.8  # Face shouldn't be more than 80% of image width
            is_face_not_too_close = bool(face_width < max_face_width)
            
            # Calculate face tilt
            eye_angle = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
            face_tilt = abs(np.degrees(eye_angle))
            is_face_straight = bool(face_tilt < 15)  # Less than 15 degrees tilt
            
            # Overall position quality
            is_good = bool(is_centered_horizontal and is_centered_vertical and 
                      is_face_size_good and is_face_not_too_close and is_face_straight)
            
            if not is_good:
                issues = []
                if not is_centered_horizontal:
                    issues.append("face_not_centered_horizontal")
                if not is_centered_vertical:
                    issues.append("face_not_centered_vertical")
                if not is_face_size_good:
                    issues.append("face_too_small")
                if not is_face_not_too_close:
                    issues.append("face_too_close")
                if not is_face_straight:
                    issues.append("face_tilted")
                quality = "position_issues"
            else:
                issues = []
                quality = "good"
            
            return {
                "is_good": bool(is_good),
                "quality": quality,
                "face_center_x": float(face_center_x),
                "face_center_y": float(face_center_y),
                "face_width": float(face_width),
                "face_tilt": float(face_tilt),
                "issues": issues,
                "is_centered_horizontal": bool(is_centered_horizontal),
                "is_centered_vertical": bool(is_centered_vertical),
                "is_face_size_good": bool(is_face_size_good),
                "is_face_straight": bool(is_face_straight)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing position: {e}")
            return {"is_good": False, "quality": "error", "error": str(e)}
    
    def analyze_stillness(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze if the face is still (not moving)"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {
                    "is_good": False,
                    "quality": "no_face_detected",
                    "message": "No face detected for stillness analysis"
                }
            
            # Get current landmarks
            current_landmarks = results.multi_face_landmarks[0]
            
            # If this is the first frame, store landmarks and return
            if self.previous_landmarks is None:
                self.previous_landmarks = current_landmarks
                return {
                    "is_good": False,
                    "quality": "initializing",
                    "message": "Initializing stillness detection"
                }
            
            # Calculate movement between frames
            movement_score = 0
            num_landmarks = len(current_landmarks.landmark)
            
            # Use key facial landmarks for movement detection
            key_landmarks = [4, 33, 263, 61, 291, 199]  # nose, eyes, mouth corners
            
            for landmark_id in key_landmarks:
                if landmark_id < num_landmarks:
                    current = current_landmarks.landmark[landmark_id]
                    previous = self.previous_landmarks.landmark[landmark_id]
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt(
                        (current.x - previous.x)**2 + 
                        (current.y - previous.y)**2
                    )
                    movement_score += distance
            
            # Average movement score
            avg_movement = movement_score / len(key_landmarks)
            
            # Determine if face is still
            is_still = bool(avg_movement < self.movement_threshold)
            
            # Update previous landmarks
            self.previous_landmarks = current_landmarks
            
            # Calculate stillness percentage (lower is better)
            stillness_percentage = max(0, 100 - (avg_movement * 1000))
            
            if is_still:
                quality = "good"
                message = "Face is still"
            else:
                quality = "moving"
                message = "Face is moving, please stay still"
            
            return {
                "is_good": bool(is_still),
                "quality": quality,
                "movement_score": float(avg_movement),
                "stillness_percentage": float(stillness_percentage),
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stillness: {e}")
            return {"is_good": False, "quality": "error", "error": str(e)}
    
    def reset_stillness_detection(self):
        """Reset stillness detection for new analysis session"""
        self.previous_landmarks = None

# Global analyzer instance
analyzer = CameraAnalyzer()

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
        
        # Perform all analyses
        lighting_analysis = analyzer.analyze_lighting(image)
        position_analysis = analyzer.analyze_position(image)
        stillness_analysis = analyzer.analyze_stillness(image)
        
        # Compile results
        results = {
            "lighting": lighting_analysis,
            "position": position_analysis,
            "stillness": stillness_analysis,
            "timestamp": str(np.datetime64('now'))
        }
        
        # Convert to JSON string and back to ensure all NumPy types are handled
        results_json = json.dumps(results, cls=NumpyEncoder)
        results_dict = json.loads(results_json)
        
        logger.info(f"Camera analysis completed successfully")
        
        return JSONResponse(content=results_dict)
        
    except Exception as e:
        logger.error(f"Error in camera analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/reset")
async def reset_analysis() -> JSONResponse:
    """Reset the analysis state (useful for starting a new session)"""
    try:
        analyzer.reset_stillness_detection()
        return JSONResponse(content={"message": "Analysis state reset successfully"})
    except Exception as e:
        logger.error(f"Error resetting analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

