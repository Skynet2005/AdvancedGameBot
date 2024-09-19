"""
/ui/main_window.py

This file defines the MainWindow class, which serves as the main window for the Adaptive Game Bot application.
It initializes the user interface, sets up the game loop, and handles the updating of the game view and performance metrics.

"""
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
from ui.control_panel import ControlPanel
from ui.game_view import GameView
from ui.performance_metrics import PerformanceMetrics
from utils.image_processing import process_game_state
import traceback

class MainWindow(QMainWindow):
    def __init__(self, game_logic):
        """
        Initializes the main window of the Adaptive Game Bot application.

        Args:
            game_logic: An instance of the game logic handler.
        """
        super().__init__()
        self.game_logic = game_logic
        self.should_exit = False
        self.init_ui()
        self.init_game_loop()

    def init_ui(self):
        """Initializes the user interface components of the main window."""
        # Set the window title
        self.setWindowTitle('Adaptive Game Bot')
        # Set the window size and position
        self.setGeometry(100, 100, 1200, 800)

        # Create the central widget
        central_widget = QWidget()
        # Set the central widget
        self.setCentralWidget(central_widget)

        # Create the main layout
        main_layout = QHBoxLayout()

        # Create the left layout
        left_layout = QVBoxLayout()
        # Create the right layout
        right_layout = QVBoxLayout()

        # Initialize the control panel
        self.control_panel = ControlPanel(self.game_logic)
        # Initialize the game view
        self.game_view = GameView()
        # Initialize the performance metrics
        self.performance_metrics = PerformanceMetrics()

        # Add control panel to the left layout
        left_layout.addWidget(self.control_panel)
        # Add performance metrics to the left layout
        left_layout.addWidget(self.performance_metrics)
        # Add game view to the right layout
        right_layout.addWidget(self.game_view)

        # Add left layout to the main layout
        main_layout.addLayout(left_layout, 1)
        # Add right layout to the main layout
        main_layout.addLayout(right_layout, 3)

        # Set the layout for the central widget
        central_widget.setLayout(main_layout)

    def init_game_loop(self):
        """Initializes the game loop timer."""
        # Create a QTimer for the game loop
        self.game_timer = QTimer(self)
        # Connect the timer to the game loop
        self.game_timer.timeout.connect(self.game_loop)
        # Start the timer with a 60 FPS interval
        self.game_timer.start(1000 // 60)

    def game_loop(self):
        """Main game loop that captures game state, processes it, and updates the UI."""
        if self.control_panel.is_bot_running():
            try:
                # Capture current game state
                game_state = self.game_logic.capture_game_screen()
                if game_state is None:
                    return
                processed_state = process_game_state(game_state)
                vision_analysis = self.game_logic.process_game_state(processed_state)
                if vision_analysis is None:
                    return

                # Get action based on current state
                action = self.game_logic.get_action(vision_analysis)
                if action is None:
                    return

                # Execute the action
                self.game_logic.execute_action(action)

                # Capture next game state
                next_game_state = self.game_logic.capture_game_screen()
                if next_game_state is None:
                    return
                next_processed_state = process_game_state(next_game_state)
                next_vision_analysis = self.game_logic.process_game_state(next_processed_state)
                if next_vision_analysis is None:
                    return

                # Calculate reward based on state transition
                reward = self.game_logic.get_reward(vision_analysis, action, next_vision_analysis)

                # Update the game logic with the new state
                self.game_logic.update(vision_analysis, action, reward, next_vision_analysis, False)

                # Update the UI with the new game state
                self.update_ui({'image': next_game_state, 'vision_analysis': next_vision_analysis}, reward)

            except Exception as e:
                print(f"Error in game loop: {e}")
                traceback.print_exc()

    def update_ui(self, game_state, reward):
        """
        Updates the UI components with the new game state and reward.

        Args:
            game_state (dict): Contains 'image' and 'vision_analysis' of the game state.
            reward (float): The reward received from the last action.
        """
        # Update the game view with the new game state image
        self.game_view.update(game_state['image'])
        # Update the performance metrics with the new reward and game state
        self.performance_metrics.update(reward, game_state)

    def closeEvent(self, event):
        """Handles the window close event to perform cleanup."""
        # Set the should_exit flag to True
        self.should_exit = True
        # Stop the game timer
        self.game_timer.stop()
        # Save the models before exiting
        self.game_logic.save_models("models/")
        # Accept the close event
        event.accept()
