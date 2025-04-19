 
# quantum_matrix_rl/envs/matrix_multiply_env.py
"""Environment for matrix multiplication algorithm discovery."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any, Union

class MatrixMultiplyDiscoveryEnv(gym.Env):
    """
    Environment for discovering efficient matrix multiplication algorithms.
    
    The agent's goal is to find a sequence of operations that computes the matrix
    product C = A × B efficiently, where A and B are input matrices.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        matrix_size: int = 2,
        max_steps: int = 15,
        step_penalty: float = 0.01,
        op_cost_penalty: float = 0.1,
        operation_costs: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the environment.
        
        Args:
            matrix_size: Size of input matrices (n×n)
            max_steps: Maximum number of steps per episode
            step_penalty: Penalty factor for each step
            op_cost_penalty: Penalty factor for operation cost
            operation_costs: Dictionary mapping operation names to costs
            render_mode: Mode for rendering the environment
        """
        self.matrix_size = matrix_size
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.op_cost_penalty = op_cost_penalty
        self.operation_costs = operation_costs or {
            "outer_product": 1.0,
            "scalar_multiply": 0.5,
            "element_update": 0.3,
            "low_rank_update": 1.5,
            "block_update": 2.0
        }
        self.render_mode = render_mode
        
        # Define operation space (5 operation types, each with parameters)
        # Parameters depend on the specific operation
        self.num_ops = len(self.operation_costs)
        param_size = self.matrix_size * self.matrix_size * 3  # Max parameters needed
        
        # Define observation space
        # State includes matrices A, B, current approximation C, and step count
        state_dim = 2 * matrix_size * matrix_size + matrix_size * matrix_size + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Define action space
        # First component selects operation, remaining components are parameters
        action_dim = 1 + param_size  # op_type + parameters
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Initialize state variables
        self.A = None
        self.B = None
        self.C = None  # Current approximation
        self.target = None  # True matrix product A×B
        self.steps = 0
        self.history = []  # Track operations performed
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Generate random matrices A and B
        self.A = self.np_random.uniform(-1.0, 1.0, (self.matrix_size, self.matrix_size))
        self.B = self.np_random.uniform(-1.0, 1.0, (self.matrix_size, self.matrix_size))
        
        # Calculate target matrix product
        self.target = np.matmul(self.A, self.B)
        
        # Initialize C with zeros
        self.C = np.zeros((self.matrix_size, self.matrix_size))
        
        # Reset step counter
        self.steps = 0
        self.history = []
        
        # Create initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action selected by the agent (operation + parameters)
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        op_type_idx = int(((action[0] + 1.0) / 2.0) * self.num_ops)
        op_type_idx = np.clip(op_type_idx, 0, self.num_ops - 1)
        op_parameters = action[1:]
        
        # Map operation index to name
        op_names = list(self.operation_costs.keys())
        op_name = op_names[op_type_idx]
        
        # Execute operation and update C
        prev_C = self.C.copy()
        op_cost = self.operation_costs[op_name]
        
        # Apply the selected operation
        self._apply_operation(op_name, op_parameters)
        
        # Record operation in history
        self.history.append({
            "op_name": op_name,
            "parameters": op_parameters,
            "cost": op_cost
        })
        
        # Calculate reward
        prev_error = np.linalg.norm(prev_C - self.target)
        current_error = np.linalg.norm(self.C - self.target)
        
        # Reward is improvement in approximation minus penalties
        improvement = prev_error - current_error
        step_penalty = self.step_penalty * self.steps
        op_penalty = self.op_cost_penalty * op_cost
        
        reward = improvement - step_penalty - op_penalty
        
        # Update step counter
        self.steps += 1
        
        # Check termination conditions
        terminated = np.isclose(current_error, 0, atol=1e-6)
        truncated = self.steps >= self.max_steps
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        # Add specific info about the step
        info.update({
            "improvement": improvement,
            "step_penalty": step_penalty,
            "op_penalty": op_penalty,
            "op_name": op_name,
            "op_cost": op_cost
        })
        
        return obs, reward, terminated, truncated, info
    
    def _apply_operation(self, op_name: str, parameters: np.ndarray) -> None:
        """
        Apply the selected operation to update the approximation matrix C.
        
        Args:
            op_name: Name of the operation to apply
            parameters: Parameters for the operation
        """
        n = self.matrix_size
        param_scaled = parameters.reshape(-1)  # Flatten parameters
        
        if op_name == "outer_product":
            # Outer product: u ⊗ v
            # Parameters: u (n), v (n)
            u_idx = 0
            v_idx = n
            
            # Scale parameters to appropriate range
            u = param_scaled[u_idx:u_idx+n] 
            v = param_scaled[v_idx:v_idx+n]
            
            # Apply outer product
            outer = np.outer(u, v)
            self.C += outer
            
        elif op_name == "scalar_multiply":
            # Scalar multiplication: α * M
            # Parameters: scalar α, matrix indices i, j
            alpha = param_scaled[0]
            i = int(((param_scaled[1] + 1) / 2) * (n - 1))
            j = int(((param_scaled[2] + 1) / 2) * (n - 1))
            
            # Apply scalar multiplication to element
            self.C[i, j] += alpha
            
        elif op_name == "element_update":
            # Update specific element: C[i,j] += value
            # Parameters: indices i, j, value
            i = int(((param_scaled[0] + 1) / 2) * (n - 1))
            j = int(((param_scaled[1] + 1) / 2) * (n - 1))
            value = param_scaled[2]
            
            # Update element
            self.C[i, j] += value
            
        elif op_name == "low_rank_update":
            # Low-rank update: C += uv^T
            # Parameters: vectors u, v (both size n)
            u_idx = 0
            v_idx = n
            
            u = param_scaled[u_idx:u_idx+n]
            v = param_scaled[v_idx:v_idx+n]
            
            # Apply low-rank update
            self.C += np.outer(u, v)
            
        elif op_name == "block_update":
            # Block update: Update a submatrix of C
            # Parameters: top-left position (i,j), values
            i = int(((param_scaled[0] + 1) / 2) * (n - 1))
            j = int(((param_scaled[1] + 1) / 2) * (n - 1))
            
            # Block size is fixed to 2x2 or the remaining size
            block_size = min(2, n - max(i, j))
            
            # Extract and scale block values
            values_start = 2
            values_end = values_start + block_size * block_size
            values = param_scaled[values_start:values_end]
            
            # Reshape values to block matrix
            block = values.reshape(block_size, block_size)
            
            # Update block
            self.C[i:i+block_size, j:j+block_size] += block
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector.
        
        Returns:
            Observation vector
        """
        # Flatten matrices A, B, C
        a_flat = self.A.flatten()
        b_flat = self.B.flatten()
        c_flat = self.C.flatten()
        
        # Normalized step count
        step_norm = np.array([self.steps / self.max_steps])
        
        # Concatenate everything
        obs = np.concatenate([a_flat, b_flat, c_flat, step_norm])
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.
        
        Returns:
            Dictionary with additional information
        """
        error = np.linalg.norm(self.C - self.target)
        total_op_cost = sum(op["cost"] for op in self.history)
        
        return {
            "error": error,
            "steps": self.steps,
            "total_op_cost": total_op_cost,
            "algorithm_cost": self.steps + total_op_cost,
            "is_correct": np.isclose(error, 0, atol=1e-6)
        }
    
    def render(self):
        """
        Render the environment state (not implemented).
        
        Returns:
            If render_mode is "rgb_array", returns the rendered image
        """
        if self.render_mode == "human":
            print(f"Step: {self.steps}/{self.max_steps}")
            print(f"Matrix A:\n{self.A}")
            print(f"Matrix B:\n{self.B}")
            print(f"Current C:\n{self.C}")
            print(f"Target (A×B):\n{self.target}")
            print(f"Error: {np.linalg.norm(self.C - self.target)}")
            print(f"Last operation: {self.history[-1] if self.history else 'None'}")
            print("-" * 40)
        
        elif self.render_mode == "rgb_array":
            # Simple visualization (not fully implemented)
            # In a real implementation, would return an RGB array
            pass