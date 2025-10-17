#!/usr/bin/env python3
#New file created

import cv2
import dlib
import numpy as np
from meta_ai_api import MetaAI
import os
import logging

# Load the detector
detector = dlib.get_frontal_face_detector()
# Load the predictor

class MetaStockAI:
    def __init__(self, access_token):
        self.api = MetaAI(access_token)
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
    def get_meta_ai_insights(self, meta_ai_text):
        
        
        try:
            response = self.api.prompt(message=meta_ai_text, new_conversation=False)
            
            return response['message']
        except Exception as e:
            logging.exception(msg=e)
            return None
        
    def calculate_facial_features(shape):
        # Calculate facial feature ratios
        eye_to_eye_distance = np.linalg.norm(np.array(shape.part(36)) - np.array(shape.part(45)))
        face_width = np.linalg.norm(np.array(shape.part(0)) - np.array(shape.part(16)))
        eye_to_eye_distance_ratio = eye_to_eye_distance / face_width
        nose_to_mouth_distance = np.linalg.norm(np.array(shape.part(30)) - np.array(shape.part(51)))
        face_height = np.linalg.norm(np.array(shape.part(8)) - np.array(shape.part(57)))
        nose_to_mouth_distance_ratio = nose_to_mouth_distance / face_height
        # Calculate facial shape analysis
        face_aspect_ratio = face_width / face_height
        # Calculate eye analysis
        eye_aspect_ratio = np.linalg.norm(np.array(shape.part(37)) - np.array(shape.part(40))) / np.linalg.norm(np.array(shape.part(36)) - np.array(shape.part(39)))
        # Calculate mouth analysis
        mouth_aspect_ratio = np.linalg.norm(np.array(shape.part(48)) - np.array(shape.part(54))) / np.linalg.norm(np.array(shape.part(51)) - np.array(shape.part(57)))
        # Calculate head pose estimation
        roll = np.arctan2(shape.part(36).y - shape.part(45).y, shape.part(36).x - shape.part(45).x)
        pitch = np.arctan2(shape.part(30).y - shape.part(8).y, shape.part(30).x - shape.part(8).x)
        yaw = np.arctan2(shape.part(0).y - shape.part(16).y, shape.part(0).x - shape.part(16).x)
        return {
            "eye_to_eye_distance_ratio": eye_to_eye_distance_ratio,
            "nose_to_mouth_distance_ratio": nose_to_mouth_distance_ratio,
            "face_aspect_ratio": face_aspect_ratio,
            "eye_aspect_ratio": eye_aspect_ratio,
            "mouth_aspect_ratio": mouth_aspect_ratio,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw
        }

    def extract_features(self, image):
        """
        Extract facial features from an image.

        Args:
        - image: The input image.

        Returns:
        - features: A NumPy array of facial features.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        # Initialize features
        features = []

        for face in faces:
            # Get facial landmarks
            landmarks = self.predictor(gray, face)

            # Calculate eye aspect ratio
            left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)])
            right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                (landmarks.part(43).x, landmarks.part(43).y),
                                (landmarks.part(44).x, landmarks.part(44).y),
                                (landmarks.part(45).x, landmarks.part(45).y),
                                (landmarks.part(46).x, landmarks.part(46).y),
                                (landmarks.part(47).x, landmarks.part(47).y)])
            eye_aspect_ratio = self.calculate_eye_aspect_ratio(left_eye, right_eye)

            # Calculate mouth aspect ratio
            mouth = np.array([(landmarks.part(48).x, landmarks.part(48).y),
                            (landmarks.part(49).x, landmarks.part(49).y),
                            (landmarks.part(50).x, landmarks.part(50).y),
                            (landmarks.part(51).x, landmarks.part(51).y),
                            (landmarks.part(52).x, landmarks.part(52).y),
                            (landmarks.part(53).x, landmarks.part(53).y),
                            (landmarks.part(54).x, landmarks.part(54).y),
                            (landmarks.part(55).x, landmarks.part(55).y),
                            (landmarks.part(56).x, landmarks.part(56).y),
                            (landmarks.part(57).x, landmarks.part(57).y),
                            (landmarks.part(58).x, landmarks.part(58).y),
                            (landmarks.part(59).x, landmarks.part(59).y)])
            mouth_aspect_ratio = self.calculate_mouth_aspect_ratio(mouth)

            # Calculate facial width-to-height ratio
            facial_width_to_height_ratio = (face.right() - face.left()) / (face.bottom() - face.top())

            # Append features
            features.append([eye_aspect_ratio, mouth_aspect_ratio, facial_width_to_height_ratio])

        return np.array(features)

    def calculate_eye_aspect_ratio(self, left_eye, right_eye):
        """
        Calculate the eye aspect ratio.

        Args:
        - left_eye: The left eye landmarks.
        - right_eye: The right eye landmarks.

        Returns:
        - eye_aspect_ratio: The eye aspect ratio.
        """
        # Calculate distances
        left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])

        # Calculate eye aspect ratio
        left_eye_aspect_ratio = left_eye_height / (2 * left_eye_width)
        right_eye_aspect_ratio = right_eye_height / (2 * right_eye_width)

        return (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

    def calculate_mouth_aspect_ratio(self, mouth):
        mouth_height = np.linalg.norm(mouth[2] - mouth[10]) + np.linalg.norm(mouth[3] - mouth[9]) + np.linalg.norm(mouth[4] - mouth[8])
        mouth_width = np.linalg.norm(mouth[0] - mouth[6])
        return mouth_height / (3 * mouth_width)


def main():
    # Sample dataset
    access_token = os.environ.get("META_AI_TOKEN")
    ai = MetaStockAI(access_token=access_token)

    # Test with a new image
    new_image = cv2.imread("profile.jpg")
    new_features = ai.calculate_facial_features()
    emotion_recognition = ai.get_meta_ai_insights(meta_ai_text=f"analyze these facial landmarks for sentiment analysis: {new_features}")
    print(emotion_recognition)
    

if __name__ == "__main__":
    main()