#!/usr/bin/env python3
#New file created
import cv2
import dlib
import numpy as np
import os
import logging

from meta_ai_api import MetaAI
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetaEmotionRecogAI:
    def __init__(self, access_token):
        self.access_token = access_token
        self.ai = MetaAI(access_token)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.feature_vectors = []
        self.features_dict = {}
        self.baseline_feature_dict = {
        "pitch": {"mean": 0, "stddev": 0},
        "yaw": {"mean": 0, "stddev": 0},
        "roll": {"mean": 0, "stddev": 0},
        "eye_to_eye_distance": {"mean": 0, "stddev": 0},
        "eye_aspect_ratio": {"mean": 0, "stddev": 0},
        "mouth_aspect_ratio": {"mean": 0, "stddev": 0},
        "face_aspect_ratio": {"mean": 0, "stddev": 0},
    }
    def get_meta_ai_insights(self, meta_ai_text):
        
        
        try:
            response = self.ai.prompt(message=meta_ai_text, new_conversation=False)
            
            return response['message']
        except Exception as e:
            logging.exception(msg=e)
            return None
        
    def calculate_left_eye(self, shape):
        left_eye = np.array([shape.part(36).x, shape.part(36).y])
        return left_eye
    
    def calculate_right_eye(self, shape):
        right_eye = np.array([shape.part(45).x, shape.part(45).y])
        return right_eye
    
    def calculate_left_face(self, shape):
        left_face = np.array([shape.part(0).x, shape.part(0).y])
        return left_face
    
    def calculate_right_face(self, shape):
        right_face = np.array([shape.part(16).x, shape.part(16).y])
        return right_face
    
    def calculate_eye_2_eye_distance(self, left_eye, right_eye):
        eye_2_eye_distance = np.linalg.norm(left_eye - right_eye)
        return eye_2_eye_distance
    
    def calculate_eye_2_eye_ratio(self, eye_to_eye_distance, face_width):
        
        return eye_to_eye_distance / face_width 
    
    def calculate_eye_aspect_ratio(self, shape):
        eye_aspect_ratio = np.linalg.norm(np.array(shape.part(37).x) - np.array(shape.part(40).y)) / np.linalg.norm(np.array(shape.part(36).x) - np.array(shape.part(39).y))
        return eye_aspect_ratio
   
    def calculate_mouth_top(self, shape):
        mouth_top = np.array([shape.part(51).x, shape.part(51).y])
        return mouth_top
    
    def calculate_mouth_bottom(self, shape):
        mouth_bottom = np.array([shape.part(57).x, shape.part(57).y])
        return mouth_bottom
    
    def calculate_mouth_left(self, shape):
        mouth_left = np.array([shape.part(48).x, shape.part(48).y])
        return mouth_left
    
    def calculate_mouth_right(self, shape):
        mouth_right = np.array([shape.part(54).x, shape.part(54).y])
        return mouth_right
    
    def calculate_mouth_height(self, mouth_top, mouth_bottom):
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        return mouth_height
    
    def calculate_mouth_width(self, mouth_left, mouth_right):
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        return mouth_width
    def calculate_mouth_aspect_ratio(self, mouth_height, mouth_width):
        return mouth_height / mouth_width
    def calculate_eyes_cycle(self):
        pass
    def calculate_eyes_cycle(self):
        pass
    
    def calculate_mouth_cycle(self):
        pass
    
    def calculate_face_width(self, shape):
        face_width = np.linalg.norm(np.array([shape.part(0).x, shape.part(0).y]) - np.array([shape.part(16).x, shape.part(16).y]))
        return face_width
    
    def calculate_face_height(self, shape):
        
        face_height = np.linalg.norm(np.array([shape.part(8).x, shape.part(8).y]) - np.array([shape.part(57).x, shape.part(57).y]))
        return face_height
    
    def calculate_face_aspect_ratio(self, face_width, face_height):
        return face_width / face_height
    
    def calculate_nose_2_mouth_distance(self, shape):
        nose_to_mouth_distance = np.linalg.norm(np.array([shape.part(30).x, shape.part(30).y]) - np.array([shape.part(51).x, shape.part(51).y]))
        return nose_to_mouth_distance
    
    def calculate_nose_distance_ratio(self, nose_to_mouth_distance, face_height):
        pass
 

    def calculate_head_pose(self, shape):
        try:
            
            # 3D model points
            model_points = np.array([
                (-165, 170, -115),  # left eye
                (165, 170, -115),  # right eye
                (0, 0, 0),  # nose tip
                (-150, -150, -125),  # left mouth corner
                (150, -150, -125)  # right mouth corner
            ])
            # 2D image points
            image_points = np.array([
                (shape.part(36).x, shape.part(36).y),  # left eye
                (shape.part(45).x, shape.part(45).y),  # right eye
                (shape.part(30).x, shape.part(30).y),  # nose tip
                (shape.part(48).x, shape.part(48).y),  # left mouth corner
                (shape.part(54).x, shape.part(54).y)  # right mouth corner
            ])
            # Check if all points are valid
            if np.any(np.isnan(image_points)):
                return None
            # Camera internals
            focal_length = 1 * 720
            center = (720 / 2, 480 / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ])
            # Distortion coefficients
            dist_coeffs = np.zeros((4, 1))
            # Solve PnP
            try:
                success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            except cv2.error:
                return None
            if success:
                # Calculate Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
                yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                return pitch, yaw, roll
            else:
                return None
        except Exception as e:
            logging.exception(msg=e)
            

    def calculate_eye_features(self, shape):
        left_eye_points = np.array([
            (shape.part(36).x, shape.part(36).y),
            (shape.part(37).x, shape.part(37).y),
            (shape.part(38).x, shape.part(38).y),
            (shape.part(39).x, shape.part(39).y),
            (shape.part(40).x, shape.part(40).y),
            (shape.part(41).x, shape.part(41).y)
        ])

        right_eye_points = np.array([
            (shape.part(42).x, shape.part(42).y),
            (shape.part(43).x, shape.part(43).y),
            (shape.part(44).x, shape.part(44).y),
            (shape.part(45).x, shape.part(45).y),
            (shape.part(46).x, shape.part(46).y),
            (shape.part(47).x, shape.part(47).y)
        ])

        # Calculate eye area
        left_eye_area = cv2.contourArea(left_eye_points)
        right_eye_area = cv2.contourArea(right_eye_points)

        # Calculate pupil-to-eye corner distances
        left_pupil_to_corner_distance = np.linalg.norm(np.array([shape.part(36).x, shape.part(36).y]) - np.array([shape.part(39).x, shape.part(39).y]))
        right_pupil_to_corner_distance = np.linalg.norm(np.array([shape.part(42).x, shape.part(42).y]) - np.array([shape.part(45).x, shape.part(45).y]))

        return left_eye_area, right_eye_area, left_pupil_to_corner_distance, right_pupil_to_corner_distance

    def calculate_mouth_features(self, shape):
        mouth_points = np.array([
            (shape.part(48).x, shape.part(48).y),
            (shape.part(49).x, shape.part(49).y),
            (shape.part(50).x, shape.part(50).y),
            (shape.part(51).x, shape.part(51).y),
            (shape.part(52).x, shape.part(52).y),
            (shape.part(53).x, shape.part(53).y),
            (shape.part(54).x, shape.part(54).y),
            (shape.part(55).x, shape.part(55).y),
            (shape.part(56).x, shape.part(56).y),
            (shape.part(57).x, shape.part(57).y),
            (shape.part(58).x, shape.part(58).y),
            (shape.part(59).x, shape.part(59).y)
        ])

        # Calculate mouth area
        mouth_area = cv2.contourArea(mouth_points)

        # Calculate mouth height and width
        mouth_height = np.linalg.norm(np.array([shape.part(51).x, shape.part(51).y]) - np.array([shape.part(57).x, shape.part(57).y]))
        mouth_width = np.linalg.norm(np.array([shape.part(48).x, shape.part(48).y]) - np.array([shape.part(54).x, shape.part(54).y]))

        return mouth_area, mouth_height, mouth_width

    def calculate_facial_features(self, shape):
        head_pose = self.calculate_head_pose(shape)
        if head_pose is None:
            # Handle the case where head pose calculation fails
            pitch, yaw, roll = 0, 0, 0
        else:
            pitch, yaw, roll = head_pose
        # Calculate other facial features
        left_eye = self.calculate_left_eye(shape)
        right_eye = self.calculate_right_eye(shape)
        eye_to_eye_distance = self.calculate_eye_2_eye_distance(left_eye, right_eye)
        eye_aspect_ratio = self.calculate_eye_aspect_ratio(shape)
        mouth_top = self.calculate_mouth_top(shape)
        mouth_bottom = self.calculate_mouth_bottom(shape)
        mouth_height = self.calculate_mouth_height(mouth_top, mouth_bottom)
        mouth_left = self.calculate_mouth_left(shape)
        mouth_right = self.calculate_mouth_right(shape)
        mouth_width = self.calculate_mouth_width(mouth_left, mouth_right)
        mouth_aspect_ratio = self.calculate_mouth_aspect_ratio(mouth_height, mouth_width)
        face_width = self.calculate_face_width(shape)
        face_height = self.calculate_face_height(shape)
        face_aspect_ratio = self.calculate_face_aspect_ratio(face_width, face_height)
        feature_vector = np.array([
            pitch,
            yaw,
            roll,
            eye_to_eye_distance,
            eye_aspect_ratio,
            mouth_aspect_ratio,
            face_aspect_ratio,
        ])
        return feature_vector

    def analyze_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        

        for face in faces:
            landmarks = self.predictor(gray, face)
            feature_vector = self.calculate_facial_features(landmarks)
            self.feature_vectors.append(feature_vector)

        return self.feature_vectors
    def emotion_recognition(self, feature_vectors):
        self.feature_dict = {
            "pitch": feature_vectors[0][0],
            "yaw": feature_vectors[0][1],
            "roll": feature_vectors[0][2],
            "eye_to_eye_distance": feature_vectors[0][3],
            "eye_aspect_ratio": feature_vectors[0][4],
            "mouth_aspect_ratio": feature_vectors[0][5],
            "face_aspect_ratio": feature_vectors[0][6],
        }
        emotion_recog = self.get_meta_ai_insights(meta_ai_text=f"analyze these facial features for emotion recognition and analysis: {self.feature_dict}")
        return emotion_recog
        

def main():
    access_token = os.environ.get("META_AI_TOKEN")
    ai = MetaEmotionRecogAI(access_token)

    image = cv2.imread("profile.jpg")
    feature_vectors = ai.analyze_image(image)
    print(ai.features_dict)
   
    emotion_recog = ai.emotion_recognition(feature_vectors)
    print(emotion_recog)
    
    # Use feature vectors for emotion recognition
    # ...

if __name__ == "__main__":
    main()
