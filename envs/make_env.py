# quantum_matrix_rl/envs/make_env.py
"""Factory function for creating and registering environments."""
import gymnasium as gym
from typing import Optional, Dict, Any

from .matrix_multiply_env import MatrixMultiplyDiscoveryEnv
from ..config import config

def make_env(env_name: str, **kwargs: Any) -> gym.Env:
    """
    Creates a Gym environment with the given name and parameters.
    
    Args:
        env_name: Name of the environment to create
        **kwargs: Additional parameters to pass to the environment
    
    Returns:
        A gym environment instance
    
    Raises:
        ValueError: If the environment name is not recognized
    """
    if env_name == "MatrixMultiplyDiscoveryEnv":
        matrix_size = kwargs.get("matrix_size", config.MATRIX_SIZE)
        max_steps = kwargs.get("max_steps", config.MAX_STEPS)
        step_penalty = kwargs.get("step_penalty", config.STEP_PENALTY)
        op_cost_penalty = kwargs.get("op_cost_penalty", config.OP_COST_PENALTY)
        operation_costs = kwargs.get("operation_costs", config.OPERATION_COSTS)
        
        return MatrixMultiplyDiscoveryEnv(
            matrix_size=matrix_size,
            max_steps=max_steps,
            step_penalty=step_penalty,
            op_cost_penalty=op_cost_penalty,
            operation_costs=operation_costs
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")