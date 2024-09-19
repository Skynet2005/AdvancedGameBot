"""
/model/reward_function.py

This file defines a RewardFunction class that uses a neural network to predict rewards based on the current state, action, and next state in a game environment. The class includes methods for calculating rewards, creating input vectors, optimizing the model with game data, and saving/loading the model.

"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class RewardFunction:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(256, 256), activation='relu', solver='adam', max_iter=1000)
        self.initialized = False

    def calculate(self, state, action, next_state):
        if not self.initialized:
            return self._default_reward(state, action, next_state)
        input_vector = self._create_input_vector(state, action, next_state)
        scaled_input = self.scaler.transform([input_vector])
        return self.model.predict(scaled_input)[0]

    def _default_reward(self, state, action, next_state):
        # Simple default reward function
        return 0  # Or any heuristic value

    def _create_input_vector(self, state, action, next_state):
        state_vector = state  # Assuming state is already a vector
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1
        next_state_vector = next_state
        return np.concatenate([state_vector, action_one_hot, next_state_vector])

    def optimize(self, game_data):
        states, actions, next_states, rewards = zip(*game_data)
        input_vectors = [self._create_input_vector(s, a, ns) for s, a, ns in zip(states, actions, next_states)]
        self.scaler.fit(input_vectors)
        scaled_inputs = self.scaler.transform(input_vectors)
        self.model.fit(scaled_inputs, rewards)
        self.initialized = True

    def save(self, path):
        import joblib
        joblib.dump((self.scaler, self.model), path)

    def load(self, path):
        import joblib
        self.scaler, self.model = joblib.load(path)
        self.initialized = True
