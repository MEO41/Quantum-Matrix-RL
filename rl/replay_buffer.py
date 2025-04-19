 
# quantum_matrix_rl/rl/replay_buffer.py
"""Replay buffer for off-policy RL algorithms."""
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional

class ReplayBuffer:
    """
    A replay buffer for storing and sampling off-policy experiences.

    Stores transitions as (state, action, reward, next_state, done).
    """
    
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 1_000_000):
        """
        Initialize the replay buffer.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            capacity: Maximum capacity of the buffer
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Numpy arrays for storing transitions
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        # Store transition at current pointer
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as torch tensors
        """
        # Sample random indices
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Extract batch
        states = torch.FloatTensor(self.states[indices])
        actions = torch.FloatTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Number of transitions currently stored
        """
        return self.size