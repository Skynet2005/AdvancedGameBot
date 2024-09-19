"""
main.py

This file serves as the entry point for the Adaptive Game Bot application. It initializes the vision model, 
allows the user to select the game area, sets up the game logic, and starts the PyQt application with a game loop 
that continuously captures and processes the game screen, determines actions, and updates the game state and UI.

"""

import sys
import os
import torch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import tkinter as tk

from ui.main_window import MainWindow
from game_logic import GameLogic
from model.vision_model import VisionModel
from utils.screen_area import ScreenAreaSelector
from utils.image_processing import process_game_state
from utils.capture_gamescreen_function import capture_game_screen
from utils.execute_action_function import execute_action

def select_game_area():
    root = tk.Tk()
    app = ScreenAreaSelector(root)
    root.mainloop()

    try:
        x1, y1, x2, y2 = app.get_selection_coordinates()
        screen_width = x2 - x1
        screen_height = y2 - y1
        return screen_width, screen_height
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

def initialize_game_logic(screen_width, screen_height, vision_model):
    game_config = {
        'screen_size': (screen_width, screen_height),
        'fps': 60,
        'actions': ['up', 'down', 'left', 'right', 'action1', 'zoom_in', 'zoom_out'],
        'state_dim': vision_model.feature_dim,
    }
    return GameLogic(game_config, vision_model)

def game_loop(main_window, game_logic):
    if main_window.control_panel.is_bot_running():
        # Capture the current game screen
        state = capture_game_screen()
        if state is None:
            print("Failed to capture game screen.")
            return
        # Process the captured game screen
        processed_state = process_game_state(state)
        vision_analysis = game_logic.process_game_state(processed_state)
        if vision_analysis is None:
            print("Failed to process game state.")
            return
        # Get the action to be performed from the game logic
        action = game_logic.get_action(vision_analysis)
        if action is None:
            print("Failed to get action.")
            return
        # Execute the action in the game
        execute_action(action)
        # Since we don't have actual game state changes, use the same state
        new_state = state
        new_processed_state = processed_state
        new_vision_analysis = vision_analysis
        # Get the reward for the performed action
        reward = game_logic.get_reward(vision_analysis, action, new_vision_analysis)
        # Update the game logic with the new state and reward
        game_logic.update(vision_analysis, action, reward, new_vision_analysis, False)
        # Update the main window with the new game state and performance metrics
        main_window.update_game_view({'image': state, 'vision_analysis': vision_analysis})
        main_window.update_performance_metrics(reward, vision_analysis)
        # Check if the episode has ended
        if game_logic.is_episode_finished():
            print(f"Episode {game_logic.episode} finished. Total reward: {game_logic.total_reward}")
            game_logic.reset_episode()

def main():
    # Initialize the vision model
    vision_model = VisionModel()
    print("Vision model initialized with pre-trained weights.")

    # Let the user select the game area
    screen_width, screen_height = select_game_area()

    # Initialize the game logic with the selected area and vision model
    game_logic = initialize_game_logic(screen_width, screen_height, vision_model)

    # Initialize the PyQt application and main window
    app = QApplication(sys.argv)
    main_window = MainWindow(game_logic)
    main_window.show()

    # Set up the game loop timer
    game_timer = QTimer()
    game_timer.timeout.connect(lambda: game_loop(main_window, game_logic))
    game_timer.start(1000 // game_logic.config['fps'])  # 60 FPS

    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
