 
# quantum_matrix_rl/quantum_circuits/layers.py
"""Parameterized quantum circuit layers."""
import torch
import torch.nn as nn
import torchquantum as tq
from typing import List, Callable, Optional

def create_variational_circuit(n_qubits: int, n_layers: int) -> nn.Module:
    """
    Create a parameterized variational quantum circuit.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of repetitions of the ansatz
        
    Returns:
        Parameterized quantum circuit as a PyTorch module
    """
    class VariationalCircuit(tq.QuantumModule):
        """
        A variational quantum circuit for quantum RL.
        
        Implements a layered circuit architecture with rotations and entanglement.
        """
        
        def __init__(self):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            
            # Create quantum device
            self.qdev = tq.QuantumDevice(n_wires=n_qubits)
            
            # Create trainable parameters for the circuit
            # Each rotation gate needs one parameter per qubit per layer
            self.params_rx = nn.ParameterList([
                nn.Parameter(torch.randn(n_qubits)) for _ in range(n_layers)
            ])
            self.params_ry = nn.ParameterList([
                nn.Parameter(torch.randn(n_qubits)) for _ in range(n_layers)
            ])
            self.params_rz = nn.ParameterList([
                nn.Parameter(torch.randn(n_qubits)) for _ in range(n_layers)
            ])
            
        def forward(self, x: torch.Tensor):
            """
            Forward pass through the quantum circuit.
            
            Args:
                x: Input tensor for encoding (batch_size, encoding_dim)
                
            Returns:
                Expectation values of measurements
            """
            bsz = x.shape[0]
            
            # Create a new device for each sample in the batch
            self.qdev.reset_states(bsz)
            
            # Encode input data into the circuit
            for i in range(min(x.shape[1], self.n_qubits)):
                tq.RX(self.qdev, wires=i, params=x[:, i])
            
            # Apply variational layers
            for layer in range(self.n_layers):
                # Apply rotation gates with trainable parameters
                for i in range(self.n_qubits):
                    tq.RX(self.qdev, wires=i, params=self.params_rx[layer][i])
                    tq.RY(self.qdev, wires=i, params=self.params_ry[layer][i])
                    tq.RZ(self.qdev, wires=i, params=self.params_rz[layer][i])
                
                # Apply entangling gates
                for i in range(self.n_qubits - 1):
                    tq.CNOT(self.qdev, wires=[i, i + 1])
                
                # Connect the last qubit back to the first (ring topology)
                if self.n_qubits > 1:
                    tq.CNOT(self.qdev, wires=[self.n_qubits - 1, 0])
            
            # Measure in the Z basis (expectation values)
            expectations = []
            for i in range(self.n_qubits):
                expectations.append(tq.expval(self.qdev, op=tq.PauliZ, wires=i))
            
            return torch.stack(expectations, dim=1)
    
    return VariationalCircuit()