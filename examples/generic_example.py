import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append('../')  # Add path to access APT_Github folder modules
from apt import AdaptiveParallelTempering
from apt_preprocessor import APT_preprocessor  # Assuming the name of the file is apt_preprocessor.py


def generate_random_J_h(N):
    """
    Generate random J (adjacency) and h matrices for a given size.

    :param N: Size of the square matrix.
    :return: Tuple containing J and h matrices.
    """
    # Generate h as a random vector of size N
    h = np.random.randn(N, 1)

    # Generate a symmetric random matrix for J
    upper_triangle_indices = np.triu_indices(N, 1)
    upper_triangle_values = np.random.randn(len(upper_triangle_indices[0]))
    J = np.zeros((N, N))
    J[upper_triangle_indices] = upper_triangle_values
    J += J.T  # Make it symmetric

    return csr_matrix(J), h


def main():
    N = 10  # Size of the random J matrix
    J, h = generate_random_J_h(N)

    # Begin preprocessing with APT
    print("\n[INFO] Starting APT preprocessing...")

    # create an APT_preprocessor instance
    apt_prep = APT_preprocessor(J.copy(), h.copy())

    # run Adaptive Parallel Tempering preprocessing
    apt_prep.run(num_sweeps_MCMC=1000, num_sweeps_read=1000, num_rng=100,
                 beta_start=0.5, alpha=1.25, sigma_E_val=1000, beta_max=64, use_hash_table=0, num_cores=8)

    print("\n[INFO] APT preprocessing complete.")

    beta_list = np.load('beta_list_python.npy')
    print(f"[INFO] Beta List: {beta_list}")

    # # uncomment if you want to manually select the inverse temperatures for the replicas from beta_list
    # startingBeta = 18
    # num_replicas = 12
    # selectedBeta = range(startingBeta, startingBeta + num_replicas)
    # beta_list = beta_list[selectedBeta]

    num_replicas = beta_list.shape[0]
    print(f"[INFO] Number of replicas: {num_replicas}")

    norm_factor = np.max(np.abs(J))
    beta_list = beta_list / norm_factor
    print(f"[INFO] Normalized Beta List: {beta_list}")

    # Initiate the main APT run
    print("\n[INFO] Starting main Adaptive Parallel Tempering process...")

    # Create an AdaptiveParallelTempering instance
    apt = AdaptiveParallelTempering(J.copy(), h.copy())

    # run Adaptive Parallel Tempering
    M, Energy = apt.run(beta_list, num_replicas=num_replicas,
                        num_sweeps_MCMC=int(1e4),
                        num_sweeps_read=int(1e3),
                        num_swap_attempts=int(1e2),
                        num_swapping_pairs=1, use_hash_table=0, num_cores=8)

    # print(M)
    print(Energy)

    print("\n[INFO] APT process complete.")


if __name__ == '__main__':
    main()
