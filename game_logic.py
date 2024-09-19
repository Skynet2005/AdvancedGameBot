
"""
# game_logic.py

This file defines the GameLogic class, which integrates various machine learning models and utilities
to manage the game state, make decisions, and adapt to new game scenarios.

"""

import numpy as np
import torch
import logging
from collections import deque
import pyautogui
import time
import os
from PIL import Image
import mss

# Import models and utilities
from model.dq_network import DQNetwork, PrioritizedReplayBuffer
from model.reward_function import RewardFunction
from model.meta_learner import MetaLearner
from model.vision_model import VisionModel
from utils.image_processing import process_game_state
from utils.replay_memory import Transition
from utils.rag_utils import retrieve_relevant_info, add_new_information
# Note: The following imports are placeholders; ensure these modules are correctly implemented
# from api.openai_helper import get_goal_suggestion
# from api.ollama_helper import get_llama_insights

class GameLogic:
    def __init__(self, game_config, vision_model):
        self.config = game_config
        self.state_dim = game_config['state_dim']
        self.action_dim = len(game_config['actions'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the policy network and target network
        self.policy_net = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update parameter

        self.reward_function = RewardFunction(self.state_dim, self.action_dim)
        self.meta_learner = MetaLearner(self.policy_net)
        self.vision_model = vision_model

        self.epsilon = 0.1
        self.update_frequency = 10
        self.steps = 0
        self.episode = 0
        self.total_reward = 0

        self.replay_buffer = PrioritizedReplayBuffer(100000)
        self.batch_size = 64  # Batch size for training

        self.experience_buffer = deque(maxlen=1000)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load game area coordinates
        self.game_area = self._load_game_area()

        # Action map for execute_action
        self.action_map = {
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'action1': 'ctrl',
            'zoom_in': ('ctrl', 'scroll_up'),
            'zoom_out': ('ctrl', 'scroll_down')
        }

    def _load_game_area(self):
        # Load game area coordinates from the saved image
        if os.path.exists("images/game_area.png"):
            try:
                with Image.open("images/game_area.png") as img:
                    x1, y1, x2, y2 = img.getbbox()
                    return (x1, y1, x2, y2)
            except Exception as e:
                self.logger.error(f"Error loading game area image: {str(e)}")
                return (0, 0, self.config['screen_size'][0], self.config['screen_size'][1])
        else:
            # Default to full screen if no area is selected
            self.logger.warning("Game area not defined. Defaulting to full screen.")
            return (0, 0, self.config['screen_size'][0], self.config['screen_size'][1])

    def process_game_state(self, processed_state):
        try:
            vision_analysis = self.vision_model.comprehensive_analysis(processed_state)
            return vision_analysis
        except Exception as e:
            self.logger.error(f"Error processing game state: {str(e)}")
            return None

    def get_action(self, vision_analysis):
        try:
            state_vector = self._convert_vision_analysis_to_vector(vision_analysis)
            return self._select_action(state_vector)
        except Exception as e:
            self.logger.error(f"Error getting action: {str(e)}")
            return None

    def _select_action(self, state_vector):
        # Select an action using epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        try:
            self.steps += 1
            self.total_reward += reward
            state_vector = self._convert_vision_analysis_to_vector(state)
            next_state_vector = self._convert_vision_analysis_to_vector(next_state)
            self.replay_buffer.push(state_vector, action, reward, next_state_vector, done)
            self.experience_buffer.append((state, action, reward, next_state, done))

            if len(self.replay_buffer) >= self.batch_size:
                self._train_model()

            if self.steps % self.update_frequency == 0:
                # Update the reward function
                self._update_reward_function(state, action, next_state, reward)
                # Generate new information and add to the knowledge base
                new_info = self._generate_new_information(state, action, reward, next_state)
                if new_info:
                    add_new_information(new_info)
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")

    def _train_model(self):
        # Update the policy network using a batch of transitions
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next Q values using the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            next_q_values_max = next_q_values.max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values_max

        # Compute loss
        loss = (self.loss_fn(current_q_values, expected_q_values) * weights).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

        # Update priorities in the replay buffer
        td_errors = abs(current_q_values - expected_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors.flatten())

        # Soft update of the target network
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _update_reward_function(self, state, action, next_state, reward):
        try:
            state_vector = self._convert_vision_analysis_to_vector(state)
            next_state_vector = self._convert_vision_analysis_to_vector(next_state)
            self.reward_function.optimize([(state_vector, action, next_state_vector, reward)])
        except Exception as e:
            self.logger.error(f"Error updating reward function: {str(e)}")

    def get_reward(self, state, action, next_state):
        try:
            state_vector = self._convert_vision_analysis_to_vector(state)
            next_state_vector = self._convert_vision_analysis_to_vector(next_state)
            return self.reward_function.calculate(state_vector, action, next_state_vector)
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0

    def is_episode_finished(self):
        # Define your condition for ending an episode
        max_steps = 1000  # Adjust this value based on the game
        if self.steps >= max_steps:
            self.logger.info(f"Episode finished after {self.steps} steps")
            return True
        return False

    def reset_episode(self):
        self.episode += 1
        self.total_reward = 0
        self.steps = 0
        self.logger.info(f"Episode {self.episode} started")

    def capture_game_screen(self):
        try:
            with mss.mss() as sct:
                x1, y1, x2, y2 = self.game_area
                monitor = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                return img[:, :, :3]  # Convert from BGRA to RGB
        except Exception as e:
            self.logger.error(f"Error capturing game screen: {str(e)}")
            return None

    def execute_action(self, action_index):
        try:
            if action_index < 0 or action_index >= self.action_dim:
                self.logger.warning(f"Invalid action index: {action_index}")
                return
            action_name = self.config['actions'][action_index]
            if action_name in self.action_map:
                key = self.action_map[action_name]
                if isinstance(key, tuple):
                    pyautogui.keyDown(key[0])
                    pyautogui.scroll(10 if key[1] == 'scroll_up' else -10)
                    pyautogui.keyUp(key[0])
                else:
                    pyautogui.keyDown(key)
                    time.sleep(0.1)
                    pyautogui.keyUp(key)
                self.logger.info(f"Executed action: {action_name}")
            else:
                self.logger.warning(f"Unknown action: {action_name}")
        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")

    def _convert_vision_analysis_to_vector(self, vision_analysis):
        try:
            features = vision_analysis['features']
            # Since we are using dummy data for objects, scene, and segmentation, we'll just use the features
            return features
        except Exception as e:
            self.logger.error(f"Error converting vision analysis to vector: {str(e)}")
            return np.zeros(self.state_dim)

    def _generate_new_information(self, state, action, reward, next_state):
        try:
            state_description = f"Game state features: {state['features']}"
            action_description = f"Action taken: {self.config['actions'][action]}"
            result_description = f"Resulting reward: {reward}"
            new_info = f"{state_description}. {action_description}. {result_description}"
            return new_info
        except Exception as e:
            self.logger.error(f"Error generating new information: {str(e)}")
            return None

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def save_models(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.policy_net.state_dict(), f"{path}/policy_net.pth")
            torch.save(self.target_net.state_dict(), f"{path}/target_net.pth")
            self.reward_function.save(f"{path}/reward_function.joblib")
            self.meta_learner.save(f"{path}/meta_learner.pth")
            # Assuming the vision model has a save method
            # self.vision_model.save(f"{path}/vision_model.pth")
            self.logger.info(f"Models successfully saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")

    def load_models(self, path):
        try:
            self.policy_net.load_state_dict(torch.load(f"{path}/policy_net.pth"))
            self.target_net.load_state_dict(torch.load(f"{path}/target_net.pth"))
            self.reward_function.load(f"{path}/reward_function.joblib")
            self.meta_learner.load(f"{path}/meta_learner.pth")
            # Assuming the vision model has a load method
            # self.vision_model.load(f"{path}/vision_model.pth")
            self.logger.info(f"Models successfully loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")

    def get_recent_experiences(self, n=100):
        return list(self.experience_buffer)[-n:]

    def adapt_to_new_game(self, support_data):
        try:
            recent_experiences = self.get_recent_experiences()
            combined_data = support_data + recent_experiences
            adapted_model = self.meta_learner.adapt(combined_data)
            self.meta_learner.fine_tune(adapted_model, recent_experiences, epochs=5)
            self.policy_net.load_state_dict(adapted_model.state_dict())
            self.logger.info("Successfully adapted to new game")
        except Exception as e:
            self.logger.error(f"Error adapting to new game: {str(e)}")

    def get_ai_insights(self, game_state):
        # Placeholder for AI insights, since APIs are not implemented
        # You can implement calls to OpenAI or other APIs here
        # For now, we'll return dummy insights
        return None, None, None, None

    def set_manual_mode(self, enabled):
        # Placeholder method for setting manual mode
        if enabled:
            self.logger.info("Manual mode enabled.")
        else:
            self.logger.info("Manual mode disabled.")
