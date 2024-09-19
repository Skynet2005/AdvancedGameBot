"""
/vision/object_detection.py

This module provides the ObjectDetector class for detecting objects in images using a pre-trained Faster R-CNN model with a ResNet-50 backbone. It also includes a function to visualize the detected objects with bounding boxes and confidence scores.

"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        # Initialize the Faster R-CNN model with a ResNet-50 backbone
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode
        self.confidence_threshold = confidence_threshold  # Set the confidence threshold for detections
        # Determine the device to run the model on (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # Move the model to the appropriate device

    def detect(self, image):
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Transform the image to a tensor and add a batch dimension
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        
        # Perform inference without gradient calculation
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extract bounding boxes, labels, and scores from the predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter detections based on confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        return boxes, labels, scores

def visualize_detections(image, boxes, labels, scores):
    import cv2
    
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Draw bounding boxes and labels on the image
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
        cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Put label and score
    
    return image