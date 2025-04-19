import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union, Any

def plot_training_curve(
    rewards: List[float], 
    window_size: int = 10,
    title: str = "Training Curve",
    save_path: Optional[str] = None,
    show: bool = True,
    additional_metrics: Optional[Dict[str, List[float]]] = None
) -> plt.Figure:
    """
    Plot the training reward curve with a moving average.
    
    Args:
        rewards: List of episode rewards
        window_size: Size of the moving average window
        title: Plot title
        save_path: Path to save the figure, if None figure is not saved
        show: Whether to display the figure
        additional_metrics: Additional metrics to plot on same figure
        
    Returns:
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot raw rewards as light points
    episodes = np.arange(1, len(rewards) + 1)
    ax.scatter(episodes, rewards, alpha=0.3, color='blue', s=5, label='_nolegend_')
    
    # Calculate and plot moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax.plot(
            np.arange(window_size, len(rewards) + 1), 
            moving_avg, 
            color='blue', 
            linewidth=2, 
            label=f'Reward (MA-{window_size})'
        )
    
    # Plot additional metrics if provided
    if additional_metrics:
        colors = ['red', 'green', 'purple', 'orange']
        for i, (name, values) in enumerate(additional_metrics.items()):
            color = colors[i % len(colors)]
            if len(values) >= window_size:
                moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                ax.plot(
                    np.arange(window_size, len(values) + 1),
                    moving_avg,
                    color=color,
                    linewidth=2,
                    label=f'{name} (MA-{window_size})'
                )
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_loss_curves(
    critic_losses: List[float],
    actor_losses: List[float],
    alpha_losses: Optional[List[float]] = None,
    window_size: int = 100,
    title: str = "Training Losses",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot SAC loss curves with moving averages.
    
    Args:
        critic_losses: List of critic loss values
        actor_losses: List of actor loss values
        alpha_losses: List of temperature parameter losses (if using adaptive alpha)
        window_size: Size of the moving average window
        title: Plot title
        save_path: Path to save the figure, if None figure is not saved
        show: Whether to display the figure
        
    Returns:
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(1, max(len(critic_losses), len(actor_losses)) + 1)
    
    # Apply moving average to smooth the curves
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot critic loss
    if len(critic_losses) >= window_size:
        critic_ma = moving_average(critic_losses, window_size)
        ax.plot(
            np.arange(window_size, len(critic_losses) + 1),
            critic_ma,
            color='blue',
            label=f'Critic Loss (MA-{window_size})'
        )
    
    # Plot actor loss
    if len(actor_losses) >= window_size:
        actor_ma = moving_average(actor_losses, window_size)
        ax.plot(
            np.arange(window_size, len(actor_losses) + 1),
            actor_ma,
            color='red',
            label=f'Actor Loss (MA-{window_size})'
        )
    
    # Plot alpha loss if provided
    if alpha_losses and len(alpha_losses) >= window_size:
        alpha_ma = moving_average(alpha_losses, window_size)
        ax.plot(
            np.arange(window_size, len(alpha_losses) + 1),
            alpha_ma,
            color='green',
            label=f'Alpha Loss (MA-{window_size})'
        )
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Use log scale for better visualization
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_matrix_error(
    errors: List[float],
    best_error: float,
    title: str = "Matrix Multiplication Error",
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = True
) -> plt.Figure:
    """
    Plot the error between discovered algorithm and optimal algorithm.
    
    Args:
        errors: List of error values over episodes/steps
        best_error: Best achieved error during training
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the figure
        log_scale: Whether to use log scale for y-axis
        
    Returns:
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(1, len(errors) + 1)
    
    # Plot error curve
    ax.plot(steps, errors, color='blue', alpha=0.7, label='Error')
    
    # Highlight best error with a dashed line
    ax.axhline(y=best_error, linestyle='--', color='red', 
              label=f'Best Error: {best_error:.6f}')
    
    ax.set_xlabel('Episode/Step')
    ax.set_ylabel('Error')
    ax.set_title(title)
    
    if log_scale and min(errors) > 0:  # Ensure no negative/zero values for log scale
        ax.set_yscale('log')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_comparison(
    classic_rewards: List[float],
    quantum_rewards: List[float],
    window_size: int = 10,
    title: str = "Classic vs Quantum SAC",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare classic and quantum agent performance.
    
    Args:
        classic_rewards: Rewards from classic SAC agent
        quantum_rewards: Rewards from quantum SAC agent
        window_size: Size of the moving average window
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the figure
        
    Returns:
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate moving averages
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot classic rewards
    episodes = np.arange(1, len(classic_rewards) + 1)
    if len(classic_rewards) >= window_size:
        classic_ma = moving_average(classic_rewards, window_size)
        ax.plot(
            np.arange(window_size, len(classic_rewards) + 1),
            classic_ma,
            color='blue',
            linewidth=2,
            label=f'Classic SAC (MA-{window_size})'
        )
    
    # Plot quantum rewards
    q_episodes = np.arange(1, len(quantum_rewards) + 1)
    if len(quantum_rewards) >= window_size:
        quantum_ma = moving_average(quantum_rewards, window_size)
        ax.plot(
            np.arange(window_size, len(quantum_rewards) + 1),
            quantum_ma,
            color='red',
            linewidth=2,
            label=f'Quantum SAC (MA-{window_size})'
        )
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig