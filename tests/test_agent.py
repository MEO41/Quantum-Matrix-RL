 
import unittest
import torch
import numpy as np
import gym
import sys
import os

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.actor import Actor
from models.critic_classic import ClassicCritic
from rl.sac_agent import SACAgent
from rl.replay_buffer import ReplayBuffer
from envs.make_env import make_env

class TestSACAgent(unittest.TestCase):
    """Test cases for SAC agent."""
    
    def setUp(self):
        """Setup test environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create environment
        self.env = make_env("MatrixMultiplyDiscoveryEnv", matrix_size=2)
        
        # Get state and action dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high[0]
        
        # Create actor and critic networks
        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_scale=self.action_high,
            hidden_dims=[256, 256],
            device=self.device
        )
        
        self.critic = ClassicCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[256, 256],
            device=self.device
        )
        
        # Create SAC agent
        self.agent = SACAgent(
            actor=self.actor,
            critic=self.critic,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            buffer_size=10000,
            initial_random_steps=1000,
            device=self.device
        )
        
        # Create a small replay buffer for testing
        self.replay_buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            buffer_size=1000,
            device=self.device
        )
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        # Check that actor and critic are set
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic)
        
        # Check replay buffer
        self.assertIsNotNone(self.agent.replay_buffer)
        self.assertEqual(self.agent.replay_buffer.buffer_size, 10000)
        
        # Check optimizers
        self.assertIsNotNone(self.agent.actor_optimizer)
        self.assertIsNotNone(self.agent.critic_optimizer)
        self.assertIsNotNone(self.agent.log_alpha_optimizer)
        
    def test_select_action(self):
        """Test action selection."""
        state = self.env.reset()
        
        # Test random action (exploration)
        self.agent.total_steps = 0  # Ensure random sampling
        action = self.agent.select_action(state)
        
        # Check action shape and bounds
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= self.env.action_space.low))
        self.assertTrue(np.all(action <= self.env.action_space.high))
        
        # Test policy action (exploitation)
        self.agent.total_steps = 2000  # Beyond random steps
        action = self.agent.select_action(state)
        
        # Check action shape and bounds
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= self.env.action_space.low))
        self.assertTrue(np.all(action <= self.env.action_space.high))
        
        # Test deterministic action (evaluation)
        action = self.agent.select_action(state, evaluate=True)
        
        # Check action shape and bounds
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= self.env.action_space.low))
        self.assertTrue(np.all(action <= self.env.action_space.high))
        
    def test_update_single(self):
        """Test single update step."""
        # Fill replay buffer with some samples
        for _ in range(10):
            state = self.env.reset()
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)
            
        # Update agent
        critic_loss, actor_loss, alpha_loss = self.agent.update(self.replay_buffer)
        
        # Check losses
        self.assertIsNotNone(critic_loss)
        self.assertIsNotNone(actor_loss)
        self.assertIsNotNone(alpha_loss)
        
    def test_update_networks(self):
        """Test target network updates."""
        # Initial parameters
        critic_params_before = [p.clone().detach() for p in self.agent.critic.parameters()]
        
        # Fill replay buffer with some samples
        for _ in range(10):
            state = self.env.reset()
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.agent.replay_buffer.add(state, action, reward, next_state, done)
            
        # Update multiple times
        for _ in range(5):
            self.agent.update(self.agent.replay_buffer)
            
        # Check target network updates
        tau = self.agent.tau
        for target_param, param in zip(self.agent.critic_target.parameters(), self.agent.critic.parameters()):
            # Target should move slightly toward online parameters
            self.assertFalse(torch.allclose(target_param, param))
            
    def test_save_load(self):
        """Test model saving and loading."""
        # Create temporary files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            tmp_path = tmp.name
            
            # Save model
            self.agent.save(tmp_path)
            
            # Create new agent with same architecture
            new_agent = SACAgent(
                actor=Actor(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    action_scale=self.action_high,
                    hidden_dims=[256, 256],
                    device=self.device
                ),
                critic=ClassicCritic(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    hidden_dims=[256, 256],
                    device=self.device
                ),
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha_lr=3e-4,
                device=self.device
            )
            
            # Load saved model
            new_agent.load(tmp_path)
            
            # Compare parameters
            for p1, p2 in zip(self.agent.actor.parameters(), new_agent.actor.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
                
            for p1, p2 in zip(self.agent.critic.parameters(), new_agent.critic.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
                
    def test_full_training_cycle(self):
        """Test a full training cycle (smoke test)."""
        episodes = 3
        max_steps = 10
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Store experience
                self.agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Update if enough samples
                if self.agent.replay_buffer.size >= self.agent.batch_size and self.agent.total_steps > self.agent.initial_random_steps:
                    critic_loss, actor_loss, alpha_loss = self.agent.update(self.agent.replay_buffer)
                
                # Transition to next state
                state = next_state
                episode_reward += reward
                step += 1
                self.agent.total_steps += 1
            
            # Basic assertion to check training is running without errors
            self.assertTrue(True)  # If we got here without exceptions, test passes