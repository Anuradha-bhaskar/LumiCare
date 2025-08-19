import cv2
import mediapipe as mp
import numpy as np
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

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
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()

            # Looser thresholds (10% pixels extreme → bad)
            over_exposed = np.sum(hist_norm[230:]) > 0.1
            under_exposed = np.sum(hist_norm[:25]) > 0.1

            contrast = std_brightness / (mean_brightness + 1e-6)

            if over_exposed:
                lighting_quality = "over_exposed"
                is_good = False
            elif under_exposed:
                lighting_quality = "under_exposed"
                is_good = False
            elif mean_brightness < 80:
                lighting_quality = "too_dark"
                is_good = False
            elif mean_brightness > 180:
                lighting_quality = "too_bright"
                is_good = False
            else:
                lighting_quality = "good"
                is_good = True

            return {
                "is_good": is_good,
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
            
            # Calculate face dimensions for skin analysis
            # Use more landmarks for better face size estimation
            # Chin (landmark 152)
            chin = landmarks.landmark[152]
            chin_x, chin_y = chin.x * width, chin.y * height
            
            # Forehead (landmark 10)
            forehead = landmarks.landmark[10]
            forehead_x, forehead_y = forehead.x * width, forehead.y * height
            
            # Calculate face height and width more accurately
            face_height = abs(forehead_y - chin_y)
            face_width = abs(right_eye_x - left_eye_x) * 3.5  # More accurate face width
            
            # Calculate face coverage percentage
            face_coverage = (face_width * face_height) / (width * height) * 100
            
            # For skin analysis, we need a much closer view
            # Face should fill 60-85% of the image for optimal skin detail detection
            min_face_coverage = 60  # Minimum face coverage percentage
            max_face_coverage = 85  # Maximum face coverage percentage
            
            is_face_size_good = bool(face_coverage > min_face_coverage)
            is_face_not_too_close = bool(face_coverage < max_face_coverage)
            
            # Debug logging
            logger.info(f"Face analysis - Width: {face_width:.1f}, Height: {face_height:.1f}, Coverage: {face_coverage:.1f}%")
            logger.info(f"Coverage check - Min: {min_face_coverage}, Max: {max_face_coverage}, Is good: {is_face_size_good}, Not too close: {is_face_not_too_close}")
            
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
                    issues.append("face_too_far")
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
                "face_height": float(face_height),
                "face_coverage": float(face_coverage),
                "face_tilt": float(face_tilt),
                "issues": issues,
                "is_centered_horizontal": bool(is_centered_horizontal),
                "is_centered_vertical": bool(is_centered_vertical),
                "is_face_size_good": bool(is_face_size_good),
                "is_face_straight": bool(is_face_straight),
                "distance_feedback": self._get_distance_feedback(face_coverage, face_width, width)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing position: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    def _get_distance_feedback(self, face_coverage: float, face_width: float, image_width: float) -> str:
        """Generate specific feedback about face distance for skin analysis"""
        if face_coverage < 30:
            return "Move much closer - your face is too far away for skin analysis"
        elif face_coverage < 50:
            return "Move closer - we need a clearer view of your skin"
        elif face_coverage < 60:
            return "Move a bit closer - almost perfect for skin analysis"
        elif face_coverage > 85:
            return "Move back slightly - your face is too close"
        elif face_coverage > 75:
            return "Perfect distance for skin analysis!"
        else:
            return "Good distance - hold this position"

    def analyze_camera_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete camera frame analysis including lighting, position, and stillness
        """
        try:
            # Perform all analyses
            lighting_analysis = self.analyze_lighting(image)
            position_analysis = self.analyze_position(image)
            stillness_analysis = self.analyze_stillness(image)
            
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
            
            return results_dict
            
        except Exception as e:
            logger.error(f"Error in camera analysis: {e}")
            raise Exception(f"Analysis failed: {str(e)}")

# Global analyzer instance
camera_analyzer = CameraAnalyzer()
