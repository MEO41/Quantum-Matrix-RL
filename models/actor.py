# quantum_matrix_rl/models/actor.py
"""Actor network for the SAC agent."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class Actor(nn.Module):
    """
    Actor network that outputs a continuous action and its log probability.
    
    The actor uses a Gaussian policy with state-dependent mean and standard deviation.
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        num_hidden_layers: int = 2
    ):
        """
        Initialize the actor network.
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            num_hidden_layers: Number of hidden layers
        """
        super(Actor, self).__init__()
        
        # Build the network architecture
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        self.network = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the actor network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (mean action, log_std)
        """
        x = self.network(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy distribution.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (sampled action, log probability)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # Sample using reparameterization trick
        x_t = normal.rsample()
        
        # Apply tanh squashing to ensure actions are in [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Calculate log probability, accounting for the tanh transformation
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh squashing correction
        # log_prob = log_prob - torch.sum(torch.log(1 - y_t.pow(2) + epsilon), dim=1, keepdim=True)
        log_prob = log_prob - torch.log(1 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to numpy array."""
        return tensor.detach().cpu().numpy()
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action given a state.
        
        Args:
            state: Input state numpy array
            deterministic: If True, return the mean action instead of sampling
            
        Returns:
            Action as numpy array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if deterministic:
                mean, _ = self.forward(state_tensor)
                return self.to_numpy(torch.tanh(mean))[0]
            else:
                action, _ = self.sample(state_tensor)
                return self.to_numpy(action)[0]