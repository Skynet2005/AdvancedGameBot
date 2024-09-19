"""
/vision/optical_flow.py

This module provides the OpticalFlow class for computing and visualizing optical flow in video frames using the Lucas-Kanade method. It also includes functionality to extract motion vectors from the computed optical flow.

"""

import cv2
import numpy as np

class OpticalFlow:
    def __init__(self):
        # Initialize previous grayscale frame and optical flow to None
        self.prev_gray = None
        self.flow = None
        
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def compute(self, frame):
        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If there is no previous frame, store the current frame and return None
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Detect good features to track in the previous frame
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        
        # If no features are found, update the previous frame and return None
        if p0 is None:
            self.prev_gray = gray
            return None
        
        # Calculate optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
        
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # Compute the flow as the difference between new and old points
        self.flow = good_new - good_old
        
        # Update the previous frame
        self.prev_gray = gray
        
        return self.flow

    def visualize(self, frame):
        # If there is no flow, return the original frame
        if self.flow is None:
            return frame
        
        # Create a mask image for drawing
        mask = np.zeros_like(frame)
            
        # Draw the tracks
        for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
        
        # Overlay the mask on the frame
        output = cv2.add(frame, mask)
        return output

    def get_motion_vector(self):
        # If there is no flow, return a zero vector
        if self.flow is None:
            return np.zeros(2)
        
        # Compute the mean motion vector
        return np.mean(self.flow, axis=0)