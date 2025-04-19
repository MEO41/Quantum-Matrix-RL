 
import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.critic_classic import ClassicCritic
from models.critic_quantum import QuantumCritic

class TestCritics(unittest.TestCase):
    """Test cases for classic and quantum critic networks."""
    
    def setUp(self):
        """Setup test environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = 8
        self.action_dim = 3
        
        # Create critic networks
        self.classic_critic = ClassicCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[256, 256],
            device=self.device
        )
        
        try:
            self.quantum_critic = QuantumCritic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                n_qubits=4,
                n_qlayers=2,
                device=self.device
            )
            self.quantum_available = True
        except ImportError:
            print("TorchQuantum not available, skipping quantum critic tests")
            self.quantum_available = False
            
    def test_classic_critic_forward(self):
        """Test classic critic forward pass."""
        batch_size = 16
        state = torch.randn(batch_size, self.state_dim, device=self.device)
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Test Q1 and Q2
        q1, q2 = self.classic_critic(state, action)
        
        # Check output shapes
        self.assertEqual(q1.shape, (batch_size, 1))
        self.assertEqual(q2.shape, (batch_size, 1))
        
        # Check that outputs are different
        self.assertFalse(torch.allclose(q1, q2))
        
    def test_classic_critic_parameter_count(self):
        """Test classic critic parameter count."""
        param_count = sum(p.numel() for p in self.classic_critic.parameters())
        print(f"Classic critic parameter count: {param_count}")
        
        # Should have parameters (approximate check)
        self.assertGreater(param_count, 1000)  # Should have substantial parameters
        
    def test_classic_critic_gradients(self):
        """Test classic critic gradients."""
        batch_size = 8
        state = torch.randn(batch_size, self.state_dim, device=self.device, requires_grad=True)
        action = torch.randn(batch_size, self.action_dim, device=self.device, requires_grad=True)
        
        q1, q2 = self.classic_critic(state, action)
        loss = q1.mean() + q2.mean()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(state.grad)
        self.assertIsNotNone(action.grad)
        
        # Check that critic parameters have gradients
        for param in self.classic_critic.parameters():
            self.assertIsNotNone(param.grad)
            
    def test_quantum_critic_forward(self):
        """Test quantum critic forward pass."""
        if not self.quantum_available:
            self.skipTest("Quantum libraries not available")
            
        batch_size = 8
        state = torch.randn(batch_size, self.state_dim, device=self.device)
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Test Q1 and Q2
        q1, q2 = self.quantum_critic(state, action)
        
        # Check output shapes
        self.assertEqual(q1.shape, (batch_size, 1))
        self.assertEqual(q2.shape, (batch_size, 1))
        
    def test_quantum_critic_parameter_count(self):
        """Test quantum critic parameter count."""
        if not self.quantum_available:
            self.skipTest("Quantum libraries not available")
            
        param_count = sum(p.numel() for p in self.quantum_critic.parameters())
        print(f"Quantum critic parameter count: {param_count}")
        
        # Should have parameters (approximate check)
        self.assertGreater(param_count, 10)  # Quantum circuits typically have fewer parameters
        
    def test_quantum_critic_gradients(self):
        """Test quantum critic gradients."""
        if not self.quantum_available:
            self.skipTest("Quantum libraries not available")
            
        batch_size = 4
        state = torch.randn(batch_size, self.state_dim, device=self.device, requires_grad=True)
        action = torch.randn(batch_size, self.action_dim, device=self.device, requires_grad=True)
        
        q1, q2 = self.quantum_critic(state, action)
        loss = q1.mean() + q2.mean()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(state.grad)
        self.assertIsNotNone(action.grad)
        
        # Check that critic parameters have gradients
        grad_exists = False
        for param in self.quantum_critic.parameters():
            if param.grad is not None:
                grad_exists = True
                break
        self.assertTrue(grad_exists)  # At least some parameters should have gradients
        
    def test_output_range(self):
        """Test that critics produce reasonable value ranges."""
        batch_size = 16
        state = torch.randn(batch_size, self.state_dim, device=self.device)
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Classic critic values should be reasonable
        q1, q2 = self.classic_critic(state, action)
        
        # Check that values are finite
        self.assertTrue(torch.all(torch.isfinite(q1)))
        self.assertTrue(torch.all(torch.isfinite(q2)))
        
        # Quantum critic values should be reasonable (if available)
        if self.quantum_available:
            q1, q2 = self.quantum_critic(state, action)
            
            # Check that values are finite
            self.assertTrue(torch.all(torch.isfinite(q1)))
            self.assertTrue(torch.all(torch.isfinite(q2)))
            
    def test_critic_detach_and_cpu(self):
        """Test critic detaching and CPU conversion."""
        batch_size = 4
        state = torch.randn(batch_size, self.state_dim, device=self.device)
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Classic critic
        with torch.no_grad():
            q1, q2 = self.classic_critic(state, action)
            
        # Move to CPU
        q1_cpu = q1.cpu().numpy()
        q2_cpu = q2.cpu().numpy()
        
        self.assertIsInstance(q1_cpu, np.ndarray)
        self.assertIsInstance(q2_cpu, np.ndarray)
        
        # Check shape
        self.assertEqual(q1_cpu.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()