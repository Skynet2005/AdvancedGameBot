"""
/utils/capture_gamescreen_function.py

This module provides a function to capture a specific area of the game screen.
It uses the mss library to capture the screen and the PIL library to load the coordinates
of the area to be captured from an image file. The captured screen is then converted to a 
numpy array and returned in RGB format.

"""

import mss
import numpy as np
from PIL import Image
import os

def capture_game_screen():
    try:
        with mss.mss() as sct:
            if not os.path.exists("images/game_area.png"):
                print("Game area not defined. Please select the game area first.")
                return None
            with Image.open("images/game_area.png") as img:
                x1, y1, x2, y2 = img.getbbox()
            monitor = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            return img[:, :, :3]
    except Exception as e:
        print(f"Error capturing game screen: {e}")
        return None
