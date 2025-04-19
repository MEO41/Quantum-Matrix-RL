 
# quantum_matrix_rl/envs/__init__.py
"""Environment module initialization."""
from .make_env import make_env
from .matrix_multiply_env import MatrixMultiplyDiscoveryEnv

__all__ = ["make_env", "MatrixMultiplyDiscoveryEnv"]