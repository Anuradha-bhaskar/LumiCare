import base64
import logging
from typing import List

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

class SkinImageProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_face_region(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Extract face region from image using landmarks"""
        height, width = image.shape[:2]
        face_oval_landmarks = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                               397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                               172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        x_coords: List[float] = []
        y_coords: List[float] = []
        for landmark_id in face_oval_landmarks:
            if landmark_id < len(landmarks.landmark):
                landmark = landmarks.landmark[landmark_id]
                x_coords.append(landmark.x * width)
                y_coords.append(landmark.y * height)
        if not x_coords:
            x_coords = [landmark.x * width for landmark in landmarks.landmark]
            y_coords = [landmark.y * height for landmark in landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        padding_x = max(20, int((x_max - x_min) * 0.1))
        padding_y = max(20, int((y_max - y_min) * 0.1))
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(width, x_max + padding_x)
        y_max = min(height, y_max + padding_y)
        return image[y_min:y_max, x_min:x_max]

    def get_under_eye_region(self, image: np.ndarray, landmarks, eye_landmarks: List[int], width: int, height: int) -> np.ndarray:
        """Extract under-eye region for dark circle analysis"""
        try:
            eye_coords: List[tuple] = []
            for landmark_id in eye_landmarks:
                if landmark_id < len(landmarks.landmark):
                    landmark = landmarks.landmark[landmark_id]
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    eye_coords.append((x, y))
            if len(eye_coords) < 3:
                return np.array([])
            x_coords = [coord[0] for coord in eye_coords]
            y_coords = [coord[1] for coord in eye_coords]
            x_min, x_max = max(0, min(x_coords)), min(width, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(height, max(y_coords))
            
            # Reduced extension - was 0.8, now 0.4
            y_extension = int((y_max - y_min) * 0.4)
            y_max = min(height, y_max + y_extension)
            
            padding = max(5, int((x_max - x_min) * 0.1))
            x_min = max(0, x_min - padding)
            x_max = min(width, x_max + padding)
            return image[y_min:y_max, x_min:x_max]
        except Exception as e:
            logger.error(f"Error extracting under-eye region: {e}")
            return np.array([])

    def encode_face_region(self, face_region: np.ndarray) -> str:
        """Encode face region as base64 string"""
        try:
            if face_region.size == 0:
                return ""
            _, buffer = cv2.imencode('.jpg', face_region, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Error encoding face region: {e}")
            return ""

    def process_image_for_analysis(self, image: np.ndarray):
        """Process image and extract face landmarks and region"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                raise Exception("No face detected in image")
            landmarks = results.multi_face_landmarks[0]
            face_region = self.extract_face_region(image, landmarks)
            return landmarks, face_region
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

# Global instance
skin_image_processor = SkinImageProcessor()
