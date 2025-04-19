import os
import time
import argparse
import numpy as np
import torch
import gym

from envs.make_env import make_env
from models.actor import Actor
from models.critic_classic import ClassicCritic
from models.critic_quantum import QuantumCritic
from rl.sac_agent import SACAgent
from rl.trainer import Trainer
from utils.logger import Logger
from utils.plots import plot_training_curve, plot_loss_curves, plot_comparison
from utils.eval import evaluate_agent, save_model_results, compare_training_times
import config

def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Matrix RL")
    parser.add_argument("--mode", type=str, default="classic", choices=["classic", "quantum", "compare"],
                        help="Run classic or quantum critic, or compare both")
    parser.add_argument("--env", type=str, default="MatrixMultiplyDiscoveryEnv",
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluate every N episodes")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--matrix_size", type=int, default=3,
                        help="Size of matrices for multiplication (NxN)")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable rendering during evaluation")
                        
    return parser.parse_args()

def set_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def run_classic_sac(args, env, eval_env=None):
    """Run training with classic SAC."""
    print(f"\n{'='*50}\nRunning Classic SAC\n{'='*50}")
    
    # Setup logger
    logger = Logger(
        log_dir=os.path.join(args.save_dir, "classic"),
        experiment_name=f"classic_sac_{args.env}_s{args.seed}"
    )
    
    # Create actor and critic networks
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_high,
        hidden_dims=config.ACTOR_HIDDEN_DIMS,
        device=args.device
    )
    
    critic = ClassicCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.CRITIC_HIDDEN_DIMS,
        device=args.device
    )
    
    # Create SAC agent
    agent = SACAgent(
        actor=actor,
        critic=critic,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.CRITIC_LR,
        alpha_lr=config.ALPHA_LR,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        tau=config.TAU,
        buffer_size=config.BUFFER_SIZE,
        initial_random_steps=config.INITIAL_RANDOM_STEPS,
        device=args.device
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        eval_env=eval_env if eval_env else env,
        agent=agent,
        logger=logger,
        max_episodes=args.episodes,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        update_after=config.UPDATE_AFTER,
        update_every=config.UPDATE_EVERY,
        num_updates=config.NUM_UPDATES,
        eval_interval=args.eval_interval,
        eval_episodes=10,
        save_dir=os.path.join(args.save_dir, "classic"),
        render_eval=not args.no_render
    )
    
    # Run training
    start_time = time.time()
    results = trainer.train()
    elapsed_time = time.time() - start_time
    
    # Final evaluation
    print("\nFinal evaluation of classic agent:")
    eval_results = evaluate_agent(
        agent=agent,
        env=eval_env if eval_env else env,
        num_episodes=20,
        render=not args.no_render,
        deterministic=True,
        verbose=True
    )
    
    # Save model and results
    save_model_results(
        agent=agent,
        results={
            **results,
            **eval_results,
            "training_time": elapsed_time,
            "mode": "classic"
        },
        save_dir=os.path.join(args.save_dir, "classic"),
        model_name="classic_sac_agent"
    )
    
    # Plot learning curves
    plot_training_curve(
        rewards=results["episode_rewards"],
        window_size=10,
        title=f"Classic SAC Training Curve - {args.env}",
        save_path=os.path.join(args.save_dir, "classic", "reward_curve.png"),
        show=False
    )
    
    plot_loss_curves(
        critic_losses=results["critic_losses"],
        actor_losses=results["actor_losses"],
        alpha_losses=results.get("alpha_losses"),
        window_size=100,
        title=f"Classic SAC Loss Curves - {args.env}",
        save_path=os.path.join(args.save_dir, "classic", "loss_curves.png"),
        show=False
    )
    
    return agent, results, elapsed_time

def run_quantum_sac(args, env, eval_env=None):
    """Run training with quantum SAC."""
    print(f"\n{'='*50}\nRunning Quantum SAC\n{'='*50}")
    
    # Setup logger
    logger = Logger(
        log_dir=os.path.join(args.save_dir, "quantum"),
        experiment_name=f"quantum_sac_{args.env}_s{args.seed}"
    )
    
    # Create actor and critic networks
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_high,
        hidden_dims=config.ACTOR_HIDDEN_DIMS,
        device=args.device
    )
    
    critic = QuantumCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        n_qubits=config.N_QUBITS,
        n_qlayers=config.N_QLAYERS,
        device=args.device
    )
    
    # Create SAC agent
    agent = SACAgent(
        actor=actor,
        critic=critic,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.QUANTUM_CRITIC_LR,  # Potentially different LR for quantum
        alpha_lr=config.ALPHA_LR,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        tau=config.TAU,
        buffer_size=config.BUFFER_SIZE,
        initial_random_steps=config.INITIAL_RANDOM_STEPS,
        device=args.device
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        eval_env=eval_env if eval_env else env,
        agent=agent,
        logger=logger,
        max_episodes=args.episodes,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        update_after=config.UPDATE_AFTER,
        update_every=config.UPDATE_EVERY,
        num_updates=config.NUM_UPDATES,
        eval_interval=args.eval_interval,
        eval_episodes=10,
        save_dir=os.path.join(args.save_dir, "quantum"),
        render_eval=not args.no_render
    )
    
    # Run training
    start_time = time.time()
    results = trainer.train()
    elapsed_time = time.time() - start_time
    
    # Final evaluation
    print("\nFinal evaluation of quantum agent:")
    eval_results = evaluate_agent(
        agent=agent,
        env=eval_env if eval_env else env,
        num_episodes=20,
        render=not args.no_render,
        deterministic=True,
        verbose=True
    )
    
    # Save model and results
    save_model_results(
        agent=agent,
        results={
            **results,
            **eval_results,
            "training_time": elapsed_time,
            "mode": "quantum"
        },
        save_dir=os.path.join(args.save_dir, "quantum"),
        model_name="quantum_sac_agent"
    )
    
    # Plot learning curves
    plot_training_curve(
        rewards=results["episode_rewards"],
        window_size=10,
        title=f"Quantum SAC Training Curve - {args.env}",
        save_path=os.path.join(args.save_dir, "quantum", "reward_curve.png"),
        show=False
    )
    
    plot_loss_curves(
        critic_losses=results["critic_losses"],
        actor_losses=results["actor_losses"],
        alpha_losses=results.get("alpha_losses"),
        window_size=100,
        title=f"Quantum SAC Loss Curves - {args.env}",
        save_path=os.path.join(args.save_dir, "quantum", "loss_curves.png"),
        show=False
    )
    
    return agent, results, elapsed_time

def compare_agents(args, classic_results, quantum_results, classic_time, quantum_time):
    """Compare classic and quantum agents."""
    print(f"\n{'='*50}\nComparing Classic vs Quantum SAC\n{'='*50}")
    
    # Create comparison directory
    compare_dir = os.path.join(args.save_dir, "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    
    # Compare training curves
    plot_comparison(
        classic_rewards=classic_results["episode_rewards"],
        quantum_rewards=quantum_results["episode_rewards"],
        window_size=10,
        title=f"Classic vs Quantum SAC - {args.env}",
        save_path=os.path.join(compare_dir, "reward_comparison.png"),
        show=True
    )
    
    # Compare training efficiency
    time_comparison = compare_training_times(
        classic_times=classic_results.get("step_times", [classic_time / len(classic_results["episode_rewards"])]),
        quantum_times=quantum_results.get("step_times", [quantum_time / len(quantum_results["episode_rewards"])]),
        classic_rewards=classic_results["episode_rewards"],
        quantum_rewards=quantum_results["episode_rewards"]
    )
    
    # Save comparison results
    comparison_results = {
        "classic_best_reward": float(max(classic_results["episode_rewards"])),
        "quantum_best_reward": float(max(quantum_results["episode_rewards"])),
        "classic_training_time": float(classic_time),
        "quantum_training_time": float(quantum_time),
        "time_comparison": time_comparison,
        "classic_final_eval": classic_results.get("avg_reward", 0),
        "quantum_final_eval": quantum_results.get("avg_reward", 0)
    }
    
    with open(os.path.join(compare_dir, "comparison_results.json"), "w") as f:
        import json
        json.dump(comparison_results, f, indent=2)
    
    print("\nComparison results:")
    print(f"  Classic SAC best reward: {comparison_results['classic_best_reward']:.4f}")
    print(f"  Quantum SAC best reward: {comparison_results['quantum_best_reward']:.4f}")
    print(f"  Classic SAC training time: {classic_time:.2f} seconds")
    print(f"  Quantum SAC training time: {quantum_time:.2f} seconds")
    print(f"  Time ratio (quantum/classic): {time_comparison['time_ratio']:.2f}x")
    
    return comparison_results

def main():
    args = parse_args()
    set_seeds(args.seed)
    
    # Create environments
    env = make_env(args.env, matrix_size=args.matrix_size)
    eval_env = make_env(args.env, matrix_size=args.matrix_size)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run based on mode
    if args.mode == "classic" or args.mode == "compare":
        classic_agent, classic_results, classic_time = run_classic_sac(args, env, eval_env)
    else:
        classic_agent, classic_results, classic_time = None, None, 0
        
    if args.mode == "quantum" or args.mode == "compare":
        quantum_agent, quantum_results, quantum_time = run_quantum_sac(args, env, eval_env)
    else:
        quantum_agent, quantum_results, quantum_time = None, None, 0
        
    if args.mode == "compare":
        comparison_results = compare_agents(
            args, 
            classic_results, 
            quantum_results, 
            classic_time, 
            quantum_time
        )
    
    print(f"\nAll experiments completed. Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()