 
# quantum_matrix_rl/rl/trainer.py
"""Training loop for SAC agent."""
import gymnasium as gym
import numpy as np
import torch
import time
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from .sac_agent import SACAgent
from .replay_buffer import ReplayBuffer
from ..utils.logger import Logger

class Trainer:
    """
    Trainer for SAC agent with support for both classical and quantum critics.
    
    Handles the training loop, evaluation, and logging.
    """
    
    def __init__(
        self,
        env: gym.Env,
        agent: SACAgent,
        replay_buffer: ReplayBuffer,
        logger: Optional[Logger] = None,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        update_interval: int = 1,
        updates_per_step: int = 1,
        eval_interval: int = 1000,
        num_eval_episodes: int = 5,
        total_steps: int = 1_000_000,
        checkpoint_interval: int = 10000,
        checkpoint_path: str = "./checkpoints",
        device: str = "cpu"
    ):
        """
        Initialize the trainer.
        
        Args:
            env: Training environment
            agent: SAC agent
            replay_buffer: Replay buffer for storing transitions
            logger: Logger for tracking metrics
            batch_size: Batch size for updates
            warmup_steps: Number of random steps for buffer warmup
            update_interval: Frequency of updates (in steps)
            updates_per_step: Number of updates per step
            eval_interval: Frequency of evaluations (in steps)
            num_eval_episodes: Number of episodes per evaluation
            total_steps: Total number of training steps
            checkpoint_interval: Frequency of checkpoints (in steps)
            checkpoint_path: Path to save checkpoints
            device: Device for computations
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.updates_per_step = updates_per_step
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.total_steps = total_steps
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Track steps and episodes
        self.steps = 0
        self.episodes = 0
        
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Dictionary with training metrics
        """
        # Initialize metrics tracking
        metrics = {
            'train_rewards': [],
            'eval_rewards': [],
            'train_steps': [],
            'train_episodes': [],
            'eval_episodes': [],
            'critic_loss': [],
            'actor_loss': [],
            'alpha': [],
            'q_mean': [],
            'logprob_mean': []
        }
        
        # Start training
        print("Starting training...")
        self.steps = 0
        self.episodes = 0
        episode_reward = 0
        episode_steps = 0
        
        # Reset environment
        state, _ = self.env.reset()
        
        # Main training loop
        with tqdm(total=self.total_steps) as pbar:
            while self.steps < self.total_steps:
                # Sampling phase
                if self.steps < self.warmup_steps:
                    # Random action during warmup
                    action = self.env.action_space.sample()
                else:
                    # Agent selects action
                    action = self.agent.select_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Add to replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.steps += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Step: {self.steps}, Episode: {self.episodes}, Reward: {episode_reward:.2f}")
                
                # Perform updates if we have enough samples
                if self.steps > self.warmup_steps and self.steps % self.update_interval == 0:
                    for _ in range(self.updates_per_step):
                        if len(self.replay_buffer) > self.batch_size:
                            # Sample from buffer
                            batch = self.replay_buffer.sample(self.batch_size)
                            # Update agent
                            update_metrics = self.agent.update_parameters(*batch)
                            
                            # Log metrics
                            if self.logger is not None:
                                self.logger.log_metrics(update_metrics, self.steps)
                            
                            # Track metrics
                            for key, value in update_metrics.items():
                                if key in metrics:
                                    metrics[key].append(value)
                
                # Handle episode completion
                if done:
                    # Reset environment
                    state, _ = self.env.reset()
                    
                    # Log episode results
                    if self.logger is not None:
                        self.logger.log_episode(
                            episode_reward, 
                            episode_steps, 
                            self.episodes,
                            self.steps
                        )
                    
                    # Track metrics
                    metrics['train_rewards'].append(episode_reward)
                    metrics['train_steps'].append(episode_steps)
                    metrics['train_episodes'].append(self.episodes)
                    
                    # Reset episode tracking
                    episode_reward = 0
                    episode_steps = 0
                    self.episodes += 1
                
                # Evaluate periodically
                if self.steps % self.eval_interval == 0:
                    eval_reward = self.evaluate()
                    metrics['eval_rewards'].append(eval_reward)
                    metrics['eval_episodes'].append(self.episodes)
                    
                    if self.logger is not None:
                        self.logger.log_evaluation(eval_reward, self.steps)
                
                # Save checkpoint periodically
                if self.steps % self.checkpoint_interval == 0:
                    self.save_checkpoint()
        
        print("Training complete!")
        # Final evaluation
        eval_reward = self.evaluate()
        metrics['eval_rewards'].append(eval_reward)
        metrics['eval_episodes'].append(self.episodes)
        
        # Final checkpoint
        self.save_checkpoint("final")
        
        return metrics
                    
    def evaluate(self) -> float:
        """
        Evaluate the agent's performance.
        
        Returns:
            Average reward over evaluation episodes
        """
        print("\nEvaluating agent...")
        eval_rewards = []
        
        for episode in range(self.num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action (deterministic for evaluation)
                action = self.agent.select_action(state, evaluate=True)
                
                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        print(f"Evaluation over {self.num_eval_episodes} episodes: {avg_reward:.2f}")
        return avg_reward
    
    def save_checkpoint(self, suffix: str = "") -> None:
        """
        Save agent checkpoint.
        
        Args:
            suffix: Additional suffix for the checkpoint name
        """
        if suffix:
            path = f"{self.checkpoint_path}/sac_agent_{suffix}.pt"
        else:
            path = f"{self.checkpoint_path}/sac_agent_step_{self.steps}.pt"
            
        self.agent.save(path)
        print(f"Checkpoint saved to {path}")