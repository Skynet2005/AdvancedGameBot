"""
/utils/execute_action_function.py

This module provides the execute_action function, which maps game actions to keyboard keys and simulates key presses using the pyautogui library. The function is used to interact with the game by executing specific actions based on the input provided.
"""

import pyautogui
import time

def execute_action(action):
    # Map game actions to keyboard keys and mouse controls
    action_map = {
        'up': 'up',
        'down': 'down',
        'left': 'left',
        'right': 'right',
        'action1': 'ctrl',
        'zoom_in': ('ctrl', 'scroll_up'),
        'zoom_out': ('ctrl', 'scroll_down')
    }

    # Check if the action is in the action map
    if action in action_map:
        key = action_map[action]
        # Press and hold the key
        pyautogui.keyDown(key)
        time.sleep(0.1)  # Hold the key for a short duration
        # Release the key
        pyautogui.keyUp(key)
    else:
        # Print a message if the action is unknown
        print(f"Unknown action: {action}")