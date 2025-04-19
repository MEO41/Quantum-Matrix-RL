 
# quantum_matrix_rl/quantum_circuits/encodings.py
"""Classical to quantum data encoding methods."""
import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from typing import List, Callable, Optional

def angle_encoding(
    qdev: tq.QuantumDevice, 
    data: torch.Tensor,
    rotation_gates: List[str] = ['rx', 'ry', 'rz']
) -> None:
    """
    Encode classical data as rotation angles.
    
    Args:
        qdev: Quantum device to apply gates to
        data: Input tensor (batch_size, n_features)
        rotation_gates: List of rotation gates to use
    """
    # Ensure the number of features doesn't exceed qubits
    n_qubits = qdev.n_wires
    n_features = min(data.shape[1], n_qubits)
    
    # Apply rotation gates according to data values
    for i in range(n_features):
        # Map feature values to [-π, π]
        angle = data[:, i] * np.pi
        
        # Apply specified rotation gates
        for gate in rotation_gates:
            if gate.lower() == 'rx':
                tq.RX(qdev, wires=i, params=angle)
            elif gate.lower() == 'ry':
                tq.RY(qdev, wires=i, params=angle)
            elif gate.lower() == 'rz':
                tq.RZ(qdev, wires=i, params=angle)

def amplitude_encoding(qdev: tq.QuantumDevice, data: torch.Tensor) -> None:
    """
    Encode classical data into quantum state amplitudes.
    
    Args:
        qdev: Quantum device to apply gates to
        data: Input tensor (batch_size, 2^n_qubits)
    """
    # Ensure data is properly normalized
    batch_size = data.shape[0]
    n_qubits = qdev.n_wires
    expected_dim = 2**n_qubits
    
    if data.shape[1] != expected_dim:
        raise ValueError(f"Input dimension should be {expected_dim} for {n_qubits} qubits")
    
    # Normalize each sample to ensure valid quantum state
    norms = torch.norm(data, dim=1, keepdim=True)
    normalized_data = data / norms
    
    # Initialize quantum state with normalized data
    # Note: This is a simplified version; a real implementation would need
    # a quantum circuit to prepare the desired amplitudes
    for i in range(batch_size):
        # In TorchQuantum, this would require custom state preparation
        # This is a placeholder for the concept
        pass