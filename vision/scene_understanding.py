"""
/vision/scene_understanding.py

This module provides the SceneUnderstanding class for analyzing and extracting features from images using a pre-trained ResNet-50 model. It also includes a function to visualize the scene understanding results.

"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SceneUnderstanding:
    def __init__(self):
        # Initialize the pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode
        # Determine the device to run the model on (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to the appropriate device
        
        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),  # Resize the image to 256x256 pixels
            transforms.CenterCrop(224),  # Center crop the image to 224x224 pixels
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])
        
        # Load ImageNet class labels
        with open('data/imagenet_classes.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def analyze(self, image):
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Transform the image and add a batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference without gradient calculation
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # Get the top 5 probabilities and their corresponding class indices
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        # Prepare the results as a list of (class_name, probability) tuples
        results = []
        for i in range(top5_prob.size(0)):
            results.append((self.classes[top5_catid[i]], top5_prob[i].item()))
        
        return results

    def get_scene_features(self, image):
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Transform the image and add a batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features from the model without gradient calculation
        with torch.no_grad():
            features = self.model.features(input_tensor)
        
        # Return the features as a numpy array
        return features.squeeze().cpu().numpy()

def visualize_scene_understanding(image, results):
    import cv2
    
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Draw the class names and probabilities on the image
    for i, (class_name, prob) in enumerate(results):
        cv2.putText(image, f"{class_name}: {prob:.2f}", (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return image