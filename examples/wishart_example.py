import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append('../')  # Add path to access APT_Github folder modules
from apt import AdaptiveParallelTempering
from apt_preprocessor import APT_preprocessor  # Assuming the name of the file is apt_preprocessor.py


def txt_to_A_wishart(txtfile):
    """
    Convert a txt file into matrices J and h.

    The txt file should contain data in the format:
    spin1 spin2 value

    If spin1 not equals spin2, the value is assigned to J (interaction).
    There is no h bias for wishart instances, so h is assigned to 0.

    Parameters:
    - txtfile (str): Path to the txt file.

    Returns:
    - tuple: Tuple containing sparse matrix J and h matrices.
    """
    W = {}

    with open(txtfile, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            x = list(map(float, line.split()))
            if int(x[0]) == int(x[1]):
                continue
            W[(int(x[0]), int(x[1]))] = x[2]
            W[(int(x[1]), int(x[0]))] = x[2]

    N = max(max(W.keys())) + 1
    h = np.zeros((N, 1))
    W_matrix = np.zeros((N, N))

    for (i, j), value in W.items():
        W_matrix[i, j] = value

    W_sparse = csr_matrix(W_matrix)

    return W_sparse, h


def main():
    size_wishart = 10  # wishart instance size
    alpha_wishart = 0.5  # alpha parameter for wishart instance
    instance = 1  # index of wishart instance
    txtfile = f'./wishart_small/wishart_planting_N_{size_wishart}_alpha_{alpha_wishart:.2f}/wishart_planting_N_{size_wishart}_alpha_{alpha_wishart:.2f}_inst_{instance}.txt'
    J, h = txt_to_A_wishart(txtfile)
    J = -J  # match the sign of Hamiltonian

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
