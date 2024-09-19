"""
# model/vision_model.py

This file defines a VisionModel class that utilizes a pre-trained EfficientNet model 
as a feature extractor. The model is capable of processing images, extracting features, 
and performing a comprehensive analysis that includes dummy data for objects, scene, 
and segmentation. The class also handles image transformations and device management 
for GPU/CPU compatibility.

"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import warnings

# Suppress deprecation warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning)

class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        # Use the pre-trained EfficientNet model with the recommended weights argument
        self.weights = models.EfficientNet_B0_Weights.DEFAULT
        self.base_model = models.efficientnet_b0(weights=self.weights)
        # Remove the classifier to use as a feature extractor
        self.base_model.classifier = nn.Identity()

        # Use the default transforms provided by the weights
        self.transform = self.weights.transforms()

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Determine the feature dimension
        self.feature_dim = self._get_feature_dim()

    def _get_feature_dim(self):
        # Pass a dummy input through the base model to get the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_output = self.base_model(dummy_input)
            return dummy_output.shape[1]

    def forward(self, x):
        # Forward pass through the base model
        return self.base_model(x)

    def comprehensive_analysis(self, processed_state):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if isinstance(processed_state, np.ndarray):
                image = Image.fromarray(
                    (processed_state * 255).astype(np.uint8))
            else:
                image = processed_state
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            # Extract features
            features = self.base_model(input_tensor).squeeze().cpu().numpy()
            # Provide dummy data for objects, scene, and segmentation
            objects = [('object', 1.0, (0, 0, 0, 0))]  # Dummy object data
            scene = [('scene', 1.0)]  # Dummy scene data
            segmentation = np.zeros((224, 224))  # Dummy segmentation
            return {
                'features': features,
                'objects': objects,
                'scene': scene,
                'segmentation': segmentation
            }
