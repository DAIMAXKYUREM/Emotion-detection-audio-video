# face_model.py
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
from typing import Union, Tuple

class FacialEmotionDetector:
    """
    A class to detect facial emotions from an image or video frame using a YOLO model.
    """
    def __init__(self, model_path='best.pt'):
        """
        Initializes the FacialEmotionDetector.

        Args:
            model_path (str): The path to the pre-trained YOLO model file.
                              Defaults to 'best.pt'.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                f"Please ensure the YOLO model is in the correct directory."
            )
        
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # --- KEY FIX ---
        # Initialize separate annotators for boxes and labels, as required by newer versions of 'supervision'.
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        print("FacialEmotionDetector initialized successfully.")

    def detect_emotion(self, frame: np.ndarray) -> Tuple[np.ndarray, Union[str, None]]:
        """
        Detects emotions in a single video frame.

        Args:
            frame (np.ndarray): The input video frame from OpenCV.

        Returns:
            A tuple containing:
                - The frame with bounding boxes and labels drawn on it.
                - The name of the most prominent detected emotion (or None if no emotion is detected).
        """
        # Perform inference on the frame
        result = self.model(frame, agnostic_nms=True)[0]
        
        # Convert the results to supervision Detections object
        detections = sv.Detections.from_ultralytics(result)
        
        # Get the label of the most confident detection
        dominant_emotion = None
        if len(detections) > 0:
            most_confident_idx = np.argmax(detections.confidence)
            dominant_emotion = detections.data['class_name'][most_confident_idx]

        # Create labels for each detection
        labels = [
            f"{self.model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]
        
        # --- KEY FIX ---
        # Annotate the frame in two steps: first draw the boxes, then draw the labels.
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        return annotated_frame, dominant_emotion

if __name__ == '__main__':
    # Example usage: Run this script to test the facial emotion detection on your webcam.
    # Make sure you have 'best.pt' in the same directory.
    
    try:
        detector = FacialEmotionDetector(model_path='best.pt')
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated_frame, emotion = detector.detect_emotion(frame)
                
                # Display the resulting frame
                cv2.imshow('Facial Emotion Detection', annotated_frame)
                
                if emotion:
                    print(f"Detected Emotion: {emotion}")

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
