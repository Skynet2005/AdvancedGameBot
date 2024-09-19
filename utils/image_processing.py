"""
utils/image_processing.py

This module contains the GameVisionDataset class for loading and transforming game images, 
and the process_game_state function for processing raw game states into a format suitable for the vision model.

"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

class GameVisionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.labels = [int(f.split('_')[0]) for f in self.images]  # Assuming filename format: "label_imagename.ext"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def process_game_state(raw_state):
    # Convert raw_state to numpy array if it's not already
    if not isinstance(raw_state, np.ndarray):
        raw_state = np.array(raw_state)
    
    # Resize the image to a fixed size (e.g., 224x224 for many pre-trained models)
    processed_state = cv2.resize(raw_state, (224, 224))
    
    # Convert to RGB if it's not already
    if len(processed_state.shape) == 2:
        processed_state = cv2.cvtColor(processed_state, cv2.COLOR_GRAY2RGB)
    elif processed_state.shape[2] == 4:
        processed_state = cv2.cvtColor(processed_state, cv2.COLOR_RGBA2RGB)
    
    # Normalize pixel values to [0, 1]
    processed_state = processed_state.astype(np.float32) / 255.0
    
    return processed_state