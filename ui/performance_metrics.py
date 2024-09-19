"""
/ui/performance_metrics.py

This module defines the PerformanceMetrics class, which is a QWidget that displays various performance metrics 
such as reward, detected objects, scene, and last action. It provides a method to update these metrics based 
on the latest game state and vision analysis.

"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class PerformanceMetrics(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Initialize and add the reward label to the layout
        self.reward_label = QLabel('Reward: 0')
        layout.addWidget(self.reward_label)

        # Initialize and add the detected objects label to the layout
        self.objects_label = QLabel('Detected Objects: None')
        layout.addWidget(self.objects_label)

        # Initialize and add the scene label to the layout
        self.scene_label = QLabel('Scene: Unknown')
        layout.addWidget(self.scene_label)

        # Initialize and add the last action label to the layout
        self.action_label = QLabel('Last Action: None')
        layout.addWidget(self.action_label)

        # Set the layout for the widget
        self.setLayout(layout)

    def update(self, reward, vision_analysis):
        # Update the reward label with the new reward value
        self.reward_label.setText(f'Reward: {reward:.2f}')

        # Extract the top 5 detected objects and update the objects label
        objects = vision_analysis['objects']
        object_names = [obj[0] for obj in objects[:5]]  # Show top 5 objects
        self.objects_label.setText(f'Detected Objects: {", ".join(object_names)}')

        # Extract the top scene prediction and update the scene label
        scene = vision_analysis['scene'][0][0]  # Get the top scene prediction
        self.scene_label.setText(f'Scene: {scene}')

        # Note: You'll need to pass the last action to this method as well
        # self.action_label.setText(f'Last Action: {last_action}')