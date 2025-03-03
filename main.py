import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, List, Optional

class MobileNetDetector:
    """
    Real-time object detection using MobileNet SSD.
    """
    
    def __init__(self, 
                 prototxt_path: str,
                 model_path: str,
                 confidence_threshold: float = 0.5):
        """
        Initialize the MobileNet SSD detector.
        
        Args:
            prototxt_path: Path to prototxt file containing model architecture
            model_path: Path to caffemodel file containing trained weights
            confidence_threshold: Minimum confidence for detection (default: 0.5)
        """
        self.confidence_threshold = confidence_threshold
        
        # Validate file paths
        for path in [prototxt_path, model_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"File not found: {path}")
        
        # Initialize the list of class labels MobileNet SSD was trained to detect
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor"]
        
        # Assign random colors to each class for visualization
        np.random.seed(42)  # For consistent colors across runs
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        
        # Load the serialized model
        try:
            print("[INFO] Loading model...")
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("[INFO] Model loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
        """
        Preprocess frame for the neural network.
        
        Args:
            frame: Input image frame
            target_size: Target size for the blob (default: 300x300)
            
        Returns:
            Preprocessed blob
        """
        return cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,  # Scale factor (1/127.5)
            size=target_size,
            mean=127.5,
            swapRB=True
        )

    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            frame: Frame with detection boxes drawn
            detections: List of (class_name, confidence, box) tuples
        """
        (h, w) = frame.shape[:2]
        blob = self.preprocess_frame(frame)
        
        # Pass the blob through the network and get detections
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Initialize our list of detection results
        results = []
        
        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get the index of the class label
                class_id = int(detections[0, 0, i, 1])
                class_name = self.CLASSES[class_id]
                
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw the detection and label on the frame
                color = self.COLORS[class_id].astype("int").tolist()
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                label = f"{class_name}: {confidence * 100:.2f}%"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                results.append((class_name, confidence, (startX, startY, endX, endY)))
        
        return frame, results

    def process_video_stream(self, source: Optional[int] = 0) -> None:
        """
        Process video stream and perform real-time object detection.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise Exception("Could not open video source")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                frame, detections = self.detect_objects(frame)
                
                # Display frame
                cv2.imshow("MobileNet SSD Detection", frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function to run real-time object detection."""
    # Model files
    prototxt_path = "./MobileNetSSD_deploy.prototxt"
    model_path = "./MobileNetSSD_deploy.caffemodel"
    
    try:
        # Initialize detector
        detector = MobileNetDetector(
            prototxt_path=prototxt_path,
            model_path=model_path,
            confidence_threshold=0.5
        )
        
        print("[INFO] Starting video stream... Press 'q' to quit.")
        detector.process_video_stream()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()