"""
ui/game_view.py

This file defines the GameView class, which is responsible for displaying the game image in the Adaptive Game Bot application.
It initializes the user interface components for the game view and provides a method to update the displayed image based on the game state.

"""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

class GameView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Initialize the QLabel to display the game image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)  # Center the image in the label
        layout = QVBoxLayout()  # Create a vertical box layout
        layout.addWidget(self.image_label)  # Add the image label to the layout
        self.setLayout(layout)  # Set the layout for the widget

    def update(self, game_state):
        # Check if the game state contains an image
        if 'image' in game_state:
            image = game_state['image']
            # Ensure the image is a numpy array
            if isinstance(image, np.ndarray):
                height, width, channel = image.shape  # Get the dimensions of the image
                bytes_per_line = 3 * width  # Calculate the number of bytes per line
                # Convert the numpy array to a QImage
                q_img = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)  # Convert the QImage to a QPixmap
                self.image_label.setPixmap(pixmap)  # Set the pixmap to the image label
