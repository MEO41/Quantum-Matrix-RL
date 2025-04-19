 
# quantum_matrix_rl/rl/sac_agent.py
"""SAC agent implementation with both classical and quantum critic."""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional, Any, Union

from ..models.actor import Actor
from ..models.critic_classic import ClassicCritic
from ..models.critic_quantum import QuantumCritic
from ..config import config

class SACAgent:
    """
    Soft Actor-Critic agent with support for both classical and quantum critics.
    
    SAC is an off-policy actor-critic algorithm that optimizes a stochastic policy
    with entropy regularization.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        learning_rate_actor: float = 3e-4,
        learning_rate_critic: float = 3e-4,
        learning_rate_alpha: float = 3e-4,
        tune_alpha: bool = True,
        use_quantum_critic: bool = False,
        num_qubits: int = 8,
        num_layers: int = 3,
        device: str = "cpu"
    ):
        """
        Initialize the SAC agent.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            num_hidden_layers: Number of hidden layers
            gamma: Discount factor
            tau: Target network update rate
            alpha: Initial entropy coefficient
            learning_rate_actor: Learning rate for the actor
            learning_rate_critic: Learning rate for the critic
            learning_rate_alpha: Learning rate for the entropy coefficient
            tune_alpha: Whether to automatically tune alpha
            use_quantum_critic: Whether to use quantum critic
            num_qubits: Number of qubits for quantum critic
            num_layers: Number of layers for quantum critic
            device: Device to use for computations
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        self.tune_alpha = tune_alpha
        self.use_quantum_critic = use_quantum_critic
        
        # Initialize actor network
        self.actor = Actor(
            obs_dim, 
            action_dim, 
            hidden_dim, 
            num_hidden_layers
        ).to(self.device)
        
        # Initialize critic networks (quantum or classical)
        if use_quantum_critic:
            self.critic = QuantumCritic(
                obs_dim, 
                action_dim, 
                hidden_dim,
                num_qubits,
                num_layers
            ).to(self.device)
            
            self.critic_target = QuantumCritic(
                obs_dim, 
                action_dim, 
                hidden_dim,
                num_qubits,
                num_layers
            ).to(self.device)
        else:
            self.critic = ClassicCritic(
                obs_dim, 
                action_dim, 
                hidden_dim, 
                num_hidden_layers
            ).to(self.device)
            
            self.critic_target = ClassicCritic(
                obs_dim, 
                action_dim, 
                hidden_dim, 
                num_hidden_layers
            ).to(self.device)
        
        # Initialize target critic with the same weights
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Freeze target critic (updated via polyak averaging)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Set up optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=learning_rate_actor
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=learning_rate_critic
        )
        
        # Set up entropy coefficient (alpha)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = alpha
        
        if tune_alpha:
            # Target entropy is -dim(A)
            self.target_entropy = -action_dim
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], 
                lr=learning_rate_alpha
            )
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action given a state.
        
        Args:
            state: Current state
            evaluate: Whether to use deterministic action (for evaluation)
            
        Returns:
            Selected action
        """
        return self.actor.act(state, deterministic=evaluate)
    
    def update_parameters(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        next_states: torch.Tensor, 
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update the agent's parameters using a batch of experiences.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Dictionary with training metrics
        """
        # Move everything to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current alpha value
        if self.tune_alpha:
            alpha = self.log_alpha.exp().item()
        else:
            alpha = self.alpha
        
        # Update critic
        with torch.no_grad():
            # Sample actions and log probs from the target policy
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * (target_q - alpha * next_log_probs)
        
        # Compute current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor loss is expectation of Q - alpha * log_prob
        actor_loss = (alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha if needed
        alpha_loss = None
        if self.tune_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        self._soft_update_target()
        
        # Return metrics
        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha,
            "q_mean": q.mean().item(),
            "logprob_mean": log_probs.mean().item()
        }
        
        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss.item()
            
        return metrics
    
    def _soft_update_target(self) -> None:
        """Soft update of target network parameters."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, path: str) -> None:
        """
        Save agent parameters to file.
        
        Args:
            path: Path to save the parameters
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha': self.alpha,
            'use_quantum_critic': self.use_quantum_critic
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load agent parameters from file.
        
        Args:
            path: Path to load the parameters from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = checkpoint['alpha']