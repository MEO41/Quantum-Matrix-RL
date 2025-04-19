 
import os
import json
import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any, Union

def evaluate_agent(
    agent,
    env,
    num_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate an agent's performance in an environment.
    
    Args:
        agent: The RL agent to evaluate
        env: The gym environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment during evaluation
        deterministic: Whether to use deterministic actions
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing evaluation metrics
    """
    rewards = []
    episode_lengths = []
    best_reward = float('-inf')
    best_actions = []
    
    for i in range(num_episodes):
        episode_reward = 0
        episode_actions = []
        done = False
        state = env.reset()
        step = 0
        
        while not done:
            # Select action
            if deterministic:
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state, evaluate=False)
                
            episode_actions.append(action.copy())
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            if render:
                env.render()
                
            episode_reward += reward
            state = next_state
            step += 1
            
            # Optional safety check for max steps
            if step >= 1000:  # Arbitrary large number to prevent infinite loops
                break
                
        rewards.append(episode_reward)
        episode_lengths.append(step)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_actions = episode_actions
            
        if verbose:
            print(f"Episode {i+1}/{num_episodes}: Reward = {episode_reward:.4f}, Steps = {step}")
            
    # Calculate metrics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_length = np.mean(episode_lengths)
    
    if verbose:
        print(f"Evaluation summary:")
        print(f"  Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"  Average episode length: {avg_length:.2f}")
        print(f"  Best episode reward: {best_reward:.4f}")
        
    return {
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "avg_length": float(avg_length),
        "best_reward": float(best_reward),
        "best_actions": best_actions,
        "all_rewards": rewards,
        "all_lengths": episode_lengths
    }

def compare_solutions(
    agent_solution: List[np.ndarray],
    reference_solution: Optional[List[np.ndarray]] = None,
    env = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare agent's solution to a reference solution.
    
    Args:
        agent_solution: List of actions representing the agent's solution
        reference_solution: List of actions for the reference solution (if available)
        env: Environment to test solutions in (if available)
        verbose: Whether to print details
        
    Returns:
        Dict with comparison metrics
    """
    results = {
        "agent_solution_length": len(agent_solution)
    }
    
    # If reference solution is provided, compare them
    if reference_solution is not None:
        results["reference_solution_length"] = len(reference_solution)
        
        # Calculate similarity score (simplified)
        matches = sum(np.array_equal(a, r) for a, r in zip(agent_solution, reference_solution))
        max_len = max(len(agent_solution), len(reference_solution))
        similarity = matches / max_len if max_len > 0 else 0
        
        results["action_match_ratio"] = similarity
        results["exact_match"] = similarity == 1.0
        
        if verbose:
            print(f"Solution comparison:")
            print(f"  Agent solution length: {len(agent_solution)}")
            print(f"  Reference solution length: {len(reference_solution)}")
            print(f"  Action match ratio: {similarity:.4f}")
            print(f"  Exact match: {similarity == 1.0}")
    
    # If environment is provided, evaluate solution performance
    if env is not None:
        # Evaluate agent solution
        state = env.reset()
        agent_total_reward = 0
        for action in agent_solution:
            next_state, reward, done, info = env.step(action)
            agent_total_reward += reward
            state = next_state
            if done:
                break
                
        results["agent_solution_reward"] = float(agent_total_reward)
        
        # Evaluate reference solution if available
        if reference_solution is not None:
            state = env.reset()
            ref_total_reward = 0
            for action in reference_solution:
                next_state, reward, done, info = env.step(action)
                ref_total_reward += reward
                state = next_state
                if done:
                    break
                    
            results["reference_solution_reward"] = float(ref_total_reward)
            results["reward_difference"] = float(agent_total_reward - ref_total_reward)
            
            if verbose:
                print(f"  Agent solution reward: {agent_total_reward:.4f}")
                print(f"  Reference solution reward: {ref_total_reward:.4f}")
                print(f"  Reward difference: {agent_total_reward - ref_total_reward:.4f}")
    
    return results

def save_model_results(
    agent,
    results: Dict[str, Any],
    save_dir: str,
    model_name: str = "model",
    save_model: bool = True
) -> None:
    """
    Save model evaluation results and optionally the model itself.
    
    Args:
        agent: The RL agent
        results: Dictionary of evaluation results
        save_dir: Directory to save results in
        model_name: Base name for saved files
        save_model: Whether to save the model weights
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save evaluation results
    results_path = os.path.join(save_dir, f"{model_name}_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            serializable_results[k] = [x.tolist() for x in v]
        else:
            serializable_results[k] = v
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save model if requested
    if save_model:
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        if hasattr(agent, 'save'):
            agent.save(model_path)
        elif hasattr(agent, 'state_dict'):
            torch.save(agent.state_dict(), model_path)
        else:
            print("Warning: Could not save model, no save method found")
            
    print(f"Results saved to {save_dir}")

def compare_training_times(
    classic_times: List[float],
    quantum_times: List[float],
    classic_rewards: List[float],
    quantum_rewards: List[float]
) -> Dict[str, Any]:
    """
    Compare training efficiency between classic and quantum approaches.
    
    Args:
        classic_times: List of elapsed times for classic agent
        quantum_times: List of elapsed times for quantum agent
        classic_rewards: List of classic agent rewards
        quantum_rewards: List of quantum agent rewards
        
    Returns:
        Dictionary of comparison metrics
    """
    classic_avg_time = np.mean(classic_times)
    quantum_avg_time = np.mean(quantum_times)
    
    classic_best_reward = np.max(classic_rewards) if classic_rewards else float('-inf')
    quantum_best_reward = np.max(quantum_rewards) if quantum_rewards else float('-inf')
    
    # Time to reach specific reward thresholds
    thresholds = [0.0, 0.5, 0.9]
    classic_times_to_threshold = {}
    quantum_times_to_threshold = {}
    
    for threshold in thresholds:
        # Classic agent
        try:
            idx = next(i for i, r in enumerate(classic_rewards) if r >= threshold)
            classic_times_to_threshold[threshold] = classic_times[idx]
        except (StopIteration, IndexError):
            classic_times_to_threshold[threshold] = float('inf')
            
        # Quantum agent
        try:
            idx = next(i for i, r in enumerate(quantum_rewards) if r >= threshold)
            quantum_times_to_threshold[threshold] = quantum_times[idx]
        except (StopIteration, IndexError):
            quantum_times_to_threshold[threshold] = float('inf')
    
    return {
        "classic_avg_time_per_step": float(classic_avg_time),
        "quantum_avg_time_per_step": float(quantum_avg_time),
        "time_ratio": float(quantum_avg_time / classic_avg_time) if classic_avg_time > 0 else float('inf'),
        "classic_best_reward": float(classic_best_reward),
        "quantum_best_reward": float(quantum_best_reward),
        "classic_times_to_threshold": {str(k): float(v) for k, v in classic_times_to_threshold.items()},
        "quantum_times_to_threshold": {str(k): float(v) for k, v in quantum_times_to_threshold.items()}
    }