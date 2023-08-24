import unittest
import numpy as np
from apt_preprocessor import APT_preprocessor
import os


class TestAPTPreprocessor(unittest.TestCase):

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
        N = 10
        self.J, self.h = self.generate_random_J_h(N)
        self.apt_preprocessor = APT_preprocessor(self.J, self.h)

    def test_initialization(self):
        """Test that the class initializes correctly."""
        self.assertIsNotNone(self.apt_preprocessor)
        self.assertTrue(np.array_equal(self.apt_preprocessor.J, self.J))
        self.assertTrue(np.array_equal(self.apt_preprocessor.h, self.h))

    def test_run_method_outputs_and_file_creation(self):
        """Test if the run method generates the expected output file."""
        if os.path.exists('beta_list_python.npy'):
            os.remove('beta_list_python.npy')  # Ensure the file is not there before the test

        beta, sigma = self.apt_preprocessor.run(num_sweeps_MCMC=10, num_sweeps_read=10, num_rng=2,
                                                beta_start=0.5, alpha=1.25, sigma_E_val=1000,
                                                beta_max=32, use_hash_table=0, num_cores=1)

        self.assertTrue(isinstance(beta, list))  # Verify beta is a list
        self.assertTrue(isinstance(sigma, list))  # Verify sigma is a list
        self.assertTrue(os.path.exists('beta_list_python.npy'))  # Verify the output file was created

    def test_valid_parameters(self):
        """Test if the parameters passed are valid."""
        with self.assertRaises(ValueError):
            self.apt_preprocessor.run(num_sweeps_MCMC=-100, num_sweeps_read=100, num_rng=2,
                                      beta_start=0.5, alpha=1.25, sigma_E_val=1000,
                                      beta_max=32, use_hash_table=0, num_cores=1)

    def tearDown(self):
        """Cleanup after tests."""
        # Remove the created file, if necessary.
        if os.path.exists('beta_list_python.npy'):
            os.remove('beta_list_python.npy')

        # Removing additional generated files and folders
        if os.path.exists('sigma_list_python.npy'):
            os.remove('sigma_list_python.npy')

        if os.path.exists('beta_sigma.png'):
            os.remove('beta_sigma.png')

        # Remove the Results directory and its contents
        if os.path.exists('Results'):
            for root, dirs, files in os.walk('Results', topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir('Results')


if __name__ == "__main__":
    unittest.main()
