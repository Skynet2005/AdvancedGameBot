""" 
/ui/control_panel.py

This file defines the ControlPanel class, which provides a user interface for controlling the game bot. 
It includes buttons and checkboxes for starting/stopping the bot, enabling manual mode, and toggling 
object detection and segmentation visualization. It also includes a slider for adjusting the exploration rate (epsilon).

"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QCheckBox, QLabel, QSlider
from PyQt5.QtCore import Qt, pyqtSignal

class ControlPanel(QWidget):
    # Signal emitted when the bot status changes
    bot_status_changed = pyqtSignal(bool)

    def __init__(self, game_logic):
        super().__init__()
        self.game_logic = game_logic
        self.bot_running = False
        self.init_ui()

    def init_ui(self):
        # Initialize the layout
        layout = QVBoxLayout()

        # Start/Stop button for the bot
        self.start_stop_button = QPushButton('Start Bot', self)
        self.start_stop_button.clicked.connect(self.toggle_bot)
        layout.addWidget(self.start_stop_button)

        # Checkbox for manual mode
        self.manual_mode_checkbox = QCheckBox('Manual Mode', self)
        self.manual_mode_checkbox.stateChanged.connect(self.toggle_manual_mode)
        layout.addWidget(self.manual_mode_checkbox)

        # Checkbox to show object detection
        self.object_detection_checkbox = QCheckBox('Show Object Detection', self)
        self.object_detection_checkbox.setChecked(True)
        layout.addWidget(self.object_detection_checkbox)

        # Checkbox to show segmentation
        self.segmentation_checkbox = QCheckBox('Show Segmentation', self)
        self.segmentation_checkbox.setChecked(True)
        layout.addWidget(self.segmentation_checkbox)

        # Slider for exploration rate (epsilon)
        layout.addWidget(QLabel('Exploration Rate (Epsilon):'))
        self.epsilon_slider = QSlider(Qt.Horizontal, self)
        self.epsilon_slider.setRange(0, 100)
        self.epsilon_slider.setValue(10)  # Default epsilon of 0.1
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)
        layout.addWidget(self.epsilon_slider)

        # Set the layout for the control panel
        self.setLayout(layout)

    def toggle_bot(self):
        # Toggle the bot running state
        self.bot_running = not self.bot_running
        self.start_stop_button.setText('Stop Bot' if self.bot_running else 'Start Bot')
        self.bot_status_changed.emit(self.bot_running)

    def toggle_manual_mode(self, state):
        # Toggle manual mode in the game logic
        self.game_logic.set_manual_mode(state == Qt.Checked)

    def update_epsilon(self):
        # Update the epsilon value in the game logic
        epsilon = self.epsilon_slider.value() / 100
        self.game_logic.set_epsilon(epsilon)

    def is_bot_running(self):
        # Check if the bot is running
        return self.bot_running

    def show_object_detection(self):
        # Check if object detection should be shown
        return self.object_detection_checkbox.isChecked()

    def show_segmentation(self):
        # Check if segmentation should be shown
        return self.segmentation_checkbox.isChecked()