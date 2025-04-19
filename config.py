 
# quantum_matrix_rl/config.py
"""Configuration for the quantum RL system."""

class Config:
    # Environment parameters
    ENV_NAME = "MatrixMultiplyDiscoveryEnv"
    MATRIX_SIZE = 2  # Size of matrices (e.g., 2x2)
    MAX_STEPS = 15  # Maximum steps per episode
    
    # RL parameters
    GAMMA = 0.99  # Discount factor
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    LEARNING_RATE_ALPHA = 3e-4
    BATCH_SIZE = 256
    BUFFER_SIZE = 1_000_000
    ALPHA = 0.2  # Initial entropy coefficient
    TARGET_UPDATE_INTERVAL = 1
    TAU = 0.005  # Soft update parameter
    
    # Network parameters
    HIDDEN_DIM = 256
    NUM_HIDDEN_LAYERS = 2
    
    # Reward parameters
    STEP_PENALTY = 0.01
    OP_COST_PENALTY = 0.1
    
    # Training parameters
    NUM_EPISODES = 10000
    EVAL_INTERVAL = 100
    LOG_INTERVAL = 10
    CHECKPOINT_INTERVAL = 500
    
    # Quantum parameters
    USE_QUANTUM_CRITIC = False  # Set to True to use quantum critic
    NUM_QUBITS = 8
    NUM_LAYERS = 3
    SHOTS = 1000  # Number of measurements for quantum circuit
    
    # Operation costs
    OPERATION_COSTS = {
        "outer_product": 1.0,
        "scalar_multiply": 0.5,
        "element_update": 0.3,
        "low_rank_update": 1.5,
        "block_update": 2.0
    }

config = Config()