 
# quantum_matrix_rl/models/critic_quantum.py
"""Quantum critic network for SAC using TorchQuantum."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import torchquantum as tq
from torchquantum.encoding import encoder_op_list_name_dict

class QuantumLayer(tq.QuantumModule):
    """
    A quantum layer using TorchQuantum.
    
    Implements a variational quantum circuit for Q-value approximation.
    """
    
    def __init__(self, n_qubits: int, n_layers: int):
        """
        Initialize the quantum layer.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of layers in the variational circuit
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.qdev = tq.QuantumDevice(n_wires=n_qubits)
        
        # Create trainable parameters for the circuit
        self.params_rx = nn.ParameterList([nn.Parameter(torch.randn(n_qubits)) for _ in range(n_layers)])
        self.params_ry = nn.ParameterList([nn.Parameter(torch.randn(n_qubits)) for _ in range(n_layers)])
        self.params_rz = nn.ParameterList([nn.Parameter(torch.randn(n_qubits)) for _ in range(n_layers)])
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the quantum layer.
        
        Args:
            x: Input tensor (encoding angles)
            
        Returns:
            Expectation values
        """
        bsz = x.shape[0]
        
        # Create a new device for each sample in the batch
        self.qdev.reset_states(bsz)
        
        # Encode input data
        for i in range(self.n_qubits):
            if i < x.shape[1]:
                tq.RX(self.qdev, wires=i, params=x[:, i])
        
        # Apply variational layers
        for layer in range(self.n_layers):
            # Apply parametrized rotation gates
            for i in range(self.n_qubits):
                tq.RX(self.qdev, wires=i, params=self.params_rx[layer][i])
                tq.RY(self.qdev, wires=i, params=self.params_ry[layer][i])
                tq.RZ(self.qdev, wires=i, params=self.params_rz[layer][i])
            
            # Apply entangling gates
            for i in range(self.n_qubits - 1):
                tq.CNOT(self.qdev, wires=[i, i + 1])
            
            # Ring connection
            if self.n_qubits > 1:
                tq.CNOT(self.qdev, wires=[self.n_qubits - 1, 0])
        
        # Measure in Z basis (expectation values)
        expectations = []
        for i in range(self.n_qubits):
            expectations.append(tq.expval(self.qdev, op=tq.PauliZ, wires=i))
        
        return torch.stack(expectations, dim=1)

class QuantumCritic(nn.Module):
    """
    Quantum critic using a hybrid quantum-classical approach.
    
    Uses two separate quantum circuits for training stability (similar to double-Q learning).
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        num_qubits: int = 8,
        num_layers: int = 3
    ):
        """
        Initialize the quantum critic.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dim: Dimension of classical hidden layers
            num_qubits: Number of qubits in the quantum circuit
            num_layers: Number of layers in the variational circuit
        """
        super(QuantumCritic, self).__init__()
        
        # Classical pre-processing
        self.pre_q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits)
        )
        
        self.pre_q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits)
        )
        
        # Quantum layers
        self.q1_quantum = QuantumLayer(num_qubits, num_layers)
        self.q2_quantum = QuantumLayer(num_qubits, num_layers)
        
        # Classical post-processing
        self.post_q1 = nn.Sequential(
            nn.Linear(num_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.post_q2 = nn.Sequential(
            nn.Linear(num_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Save configuration
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both quantum critics.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Tuple of (Q1 value, Q2 value)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Pre-processing
        x1 = self.pre_q1(x)
        x2 = self.pre_q2(x)
        
        # Quantum processing
        q1_out = self.q1_quantum(x1)
        q2_out = self.q2_quantum(x2)
        
        # Post-processing
        q1_value = self.post_q1(q1_out)
        q2_value = self.post_q2(q2_out)
        
        return q1_value, q2_value
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through only the first quantum critic.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Q1 value
        """
        x = torch.cat([state, action], dim=1)
        x1 = self.pre_q1(x)
        q1_out = self.q1_quantum(x1)
        q1_value = self.post_q1(q1_out)
        return q1_value