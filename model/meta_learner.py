"""
/model/meta_learner.py

This file contains the implementation of a MetaLearner class for meta-learning.
The MetaLearner class is designed to adapt a given model to new tasks using a
two-loop optimization process: an inner loop for task-specific adaptation and
an outer loop for meta-update across multiple tasks. The class includes methods
for adapting the model, performing meta-updates, computing loss, and saving/loading
the model state.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

class MetaLearner:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001, first_order=False):
        """
        Initialize the MetaLearner with the given model and hyperparameters.
        
        Args:
            model (nn.Module): The model to be used for meta-learning.
            lr_inner (float): Learning rate for the inner loop (adaptation).
            lr_outer (float): Learning rate for the outer loop (meta-update).
            first_order (bool): If True, use first-order approximation.
        """
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.first_order = first_order
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)
        
    def adapt(self, support_data, num_inner_steps=5):
        """
        Adapt the model to the support data using the inner loop optimization.
        
        Args:
            support_data (tuple): Data used for adaptation.
            num_inner_steps (int): Number of inner loop steps.
        
        Returns:
            nn.Module: The adapted model.
        """
        adapted_model = deepcopy(self.model)
        adapted_model.train()
        
        for _ in range(num_inner_steps):
            loss = self._compute_loss(adapted_model, support_data)
            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=not self.first_order)
            
            for param, grad in zip(adapted_model.parameters(), grads):
                if self.first_order:
                    param.data.sub_(self.lr_inner * grad.data)
                else:
                    param.data.sub_(self.lr_inner * grad)
        
        return adapted_model
    
    def meta_update(self, task_batch):
        """
        Perform a meta-update using a batch of tasks.
        
        Args:
            task_batch (list): A batch of tasks, each containing support and query data.
        
        Returns:
            float: The meta-loss value.
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0
        
        for support_data, query_data in task_batch:
            adapted_model = self.adapt(support_data)
            task_loss = self._compute_loss(adapted_model, query_data)
            meta_loss += task_loss
        
        meta_loss /= len(task_batch)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _compute_loss(self, model, data):
        """
        Compute the loss for the given model and data.
        
        Args:
            model (nn.Module): The model to compute the loss for.
            data (tuple): The data used to compute the loss.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        states, actions, rewards, next_states, dones = data
        q_values = model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = model(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.model.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, targets)
        return loss
    
    def save(self, path):
        """
        Save the model and optimizer state to the given path.
        
        Args:
            path (str): The path to save the state.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """
        Load the model and optimizer state from the given path.
        
        Args:
            path (str): The path to load the state from.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])