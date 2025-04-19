 
import numpy as np
import unittest
import gym
import sys
import os

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.make_env import make_env
from envs.matrix_multiply_env import MatrixMultiplyDiscoveryEnv

class TestMatrixMultiplyEnv(unittest.TestCase):
    """Test cases for the matrix multiplication discovery environment."""
    
    def setUp(self):
        """Setup test environment."""
        self.env = make_env("MatrixMultiplyDiscoveryEnv", matrix_size=2)
        self.env_3x3 = make_env("MatrixMultiplyDiscoveryEnv", matrix_size=3)
        
    def test_env_creation(self):
        """Test environment creation and spaces."""
        # Test observation space
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        self.assertEqual(self.env.observation_space.shape[0], 8)  # 2x2 matrices: 2x2x2 = 8
        
        # Test action space
        self.assertIsInstance(self.env.action_space, gym.spaces.Box)
        self.assertEqual(self.env.action_space.shape[0], 3)  # Index i, j and operation value
        
        # Test 3x3 matrix env
        self.assertEqual(self.env_3x3.observation_space.shape[0], 18)  # 3x3 matrices: 2x3x3 = 18
        
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        # Check state dimensions
        self.assertEqual(len(state), 8)
        
        # Check that matrices are initialized
        self.assertIsNotNone(self.env.matrix_a)
        self.assertIsNotNone(self.env.matrix_b)
        self.assertIsNotNone(self.env.matrix_c)
        
        # Check shape of matrices
        self.assertEqual(self.env.matrix_a.shape, (2, 2))
        self.assertEqual(self.env.matrix_b.shape, (2, 2))
        self.assertEqual(self.env.matrix_c.shape, (2, 2))
        
    def test_step(self):
        """Test environment step function."""
        self.env.reset()
        
        # Valid action: operate on matrix C[0, 0]
        action = np.array([0, 0, 1.0])  # Set C[0,0] to 1.0
        next_state, reward, done, info = self.env.step(action)
        
        # Check state and matrix updates
        self.assertEqual(self.env.matrix_c[0, 0], 1.0)
        self.assertEqual(len(next_state), 8)
        self.assertFalse(done)  # Should not be done after one step
        
        # Test invalid action handling (out of bounds)
        self.env.reset()
        action = np.array([10, 10, 1.0])  # Invalid indices
        next_state, reward, done, info = self.env.step(action)
        
        # Should clip indices to valid range
        self.assertTrue(0 <= self.env.last_i < 2)
        self.assertTrue(0 <= self.env.last_j < 2)
        
    def test_reward_calculation(self):
        """Test reward calculation based on matrix error."""
        self.env.reset()
        
        # Set expected matrix C = A*B
        a, b = self.env.matrix_a, self.env.matrix_b
        expected_c = np.matmul(a, b)
        
        # Start with all zeros in C
        initial_error = self.env.calculate_error()
        self.assertGreater(initial_error, 0)
        
        # Set C to the correct value at one position
        i, j = 0, 0
        correct_value = expected_c[i, j]
        action = np.array([i, j, correct_value])
        _, reward, _, _ = self.env.step(action)
        
        # Error should decrease
        new_error = self.env.calculate_error()
        self.assertLess(new_error, initial_error)
        
        # Reward should be positive
        self.assertGreater(reward, 0)
        
    def test_done_condition(self):
        """Test environment termination condition."""
        self.env.reset()
        
        # Get correct result matrix
        a, b = self.env.matrix_a, self.env.matrix_b
        correct_c = np.matmul(a, b)
        
        # Set C to the correct values one by one
        done = False
        for i in range(2):
            for j in range(2):
                action = np.array([i, j, correct_c[i, j]])
                _, reward, done, _ = self.env.step(action)
                
        # When all values are correct, done should be True
        self.assertTrue(done)
        self.assertLess(self.env.calculate_error(), self.env.error_threshold)
        
    def test_render(self):
        """Test render function (limited test)."""
        # Just ensure render doesn't crash
        self.env.reset()
        # This should return a string representation for terminal
        render_output = self.env.render(mode='ansi')
        self.assertIsInstance(render_output, str)
        
    def test_maximum_steps(self):
        """Test maximum steps limit."""
        self.env.reset()
        self.env.max_steps = 5  # Set a small max steps
        
        done = False
        steps = 0
        while not done and steps < 10:  # Safety limit
            action = self.env.action_space.sample()
            _, _, done, info = self.env.step(action)
            steps += 1
            
        # Should terminate after max_steps
        self.assertTrue(done)
        self.assertEqual(steps, 5)  # Should reach max_steps
        self.assertTrue(info.get('timeout', False))  # Info should indicate timeout

if __name__ == '__main__':
    unittest.main()