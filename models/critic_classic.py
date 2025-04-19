 
# quantum_matrix_rl/models/critic_classic.py
"""Classical critic network for SAC."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ClassicCritic(nn.Module):
    """
    Classical critic network that approximates the Q-value function.
    
    In SAC, we use two Q-networks for training stability.
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        num_hidden_layers: int = 2
    ):
        """
        Initialize the critic network.
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            num_hidden_layers: Number of hidden layers
        """
        super(ClassicCritic, self).__init__()
        
        # Build Q1 architecture
        q1_layers = [nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_hidden_layers - 1):
            q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            
        q1_layers.append(nn.Linear(hidden_dim, 1))
        
        self.q1 = nn.Sequential(*q1_layers)
        
        # Build Q2 architecture (different initialization for diversity)
        q2_layers = [nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_hidden_layers - 1):
            q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            
        q2_layers.append(nn.Linear(hidden_dim, 1))
        
        self.q2 = nn.Sequential(*q2_layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critic networks.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Tuple of (Q1 value, Q2 value)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Forward through both Q-networks
        q1_value = self.q1(x)
        q2_value = self.q2(x)
        
        return q1_value, q2_value
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through only the first critic network.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Q1 value
        """
        x = torch.cat([state, action], dim=1)
        return self.q1(x)