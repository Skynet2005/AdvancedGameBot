"""
/utils/replay_memory.py

This module defines the ReplayMemory class, which is used to store and sample transitions 
(state, action, reward, next_state, done) for training reinforcement learning models.

"""

import random
from collections import namedtuple, deque

# Define a named tuple to represent a single transition in our environment
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        # Initialize the replay memory with a fixed capacity
        self.capacity = capacity
        # Use a deque to store the transitions, with a maximum length of capacity
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        # Save a transition to the replay memory
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        # Randomly sample a batch of transitions from the replay memory
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        # Return the current size of the replay memory
        return len(self.memory)
