
"""
# model/dq_network.py

This file defines a Deep Q-Network (DQN) model for reinforcement learning. The DQNetwork class
inherits from PyTorch's nn.Module and includes methods for forward propagation, action selection,
and model updates. The network architecture consists of a feature extractor, an advantage stream,
and a value stream.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random

# Define a named tuple to store transitions in the replay buffer
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity  # Maximum size of the buffer
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment  # Increment for beta
        self.buffer = []  # The actual buffer to store transitions
        # Array to store priorities
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0  # Current position in the buffer

    def push(self, *args):
        # Add a new transition to the buffer
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Sample a batch of transitions from the buffer
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1., self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        # Update the priorities of sampled transitions
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)  # Return the current size of the buffer


class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNetwork, self).__init__()
        self.state_dim = state_dim  # Dimension of the state space
        self.action_dim = action_dim  # Dimension of the action space

        # Define the feature extractor network
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Define the advantage stream network
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Define the value stream network
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Device configuration
        self.to(self.device)

    def forward(self, state):
        # Forward pass through the network
        features = self.feature_extractor(state)
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def save(self, path):
        # Save the model parameters to a file
        torch.save(self.state_dict(), path)

    def load(self, path):
        # Load the model parameters from a file
        self.load_state_dict(torch.load(path))
        self.eval()
