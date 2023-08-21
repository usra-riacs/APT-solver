import unittest
import numpy as np
from apt import AdaptiveParallelTempering
import os


class TestAPT(unittest.TestCase):

    def generate_random_J_h(self, N):
        """Generate random J (adjacency) and h matrices for a given size."""
        h = np.random.randn(N, 1)
        upper_triangle_indices = np.triu_indices(N, 1)
        upper_triangle_values = np.random.randn(len(upper_triangle_indices[0]))
        J = np.zeros((N, N))
        J[upper_triangle_indices] = upper_triangle_values
        J += J.T  # Make it symmetric
        return J, h

    def setUp(self):
        """Setup mock data for tests."""
        self.N = 10
        self.J, self.h = self.generate_random_J_h(self.N)
        self.apt = AdaptiveParallelTempering(self.J, self.h)

    def test_initialization(self):
        """Test that the class initializes correctly."""
        self.assertIsNotNone(self.apt)
        self.assertTrue(np.array_equal(self.apt.J, self.J))
        self.assertTrue(np.array_equal(self.apt.h, self.h))

    def test_run_method(self):
        """Test if the run method generates M and Energy."""
        beta_list = np.array([0.5, 1.0, 1.5, 2.0])  # A basic list; you can adjust
        num_replicas = 4  # Matches the length of beta_list

        M, Energy = self.apt.run(beta_list, num_replicas=num_replicas,
                                 num_sweeps_MCMC=int(1e2),
                                 num_sweeps_read=int(1e2),
                                 num_swap_attempts=int(1e1),
                                 num_swapping_pairs=1, use_hash_table=0, num_cores=1)

        # Basic check if the method runs without exceptions and returns expected shapes
        self.assertEqual(M.shape, (self.N * num_replicas, self.apt.num_sweeps_MCMC))
        self.assertEqual(Energy.shape, (num_replicas,))

    def tearDown(self):
        """Cleanup after tests."""
        if os.path.exists('APT_energy.png'):
            os.remove('APT_energy.png')


if __name__ == "__main__":
    unittest.main()
