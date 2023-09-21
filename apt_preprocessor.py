import numpy as np
import os
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from cachetools import LRUCache
from concurrent.futures import ProcessPoolExecutor, as_completed


class APT_preprocessor:
    def __init__(self, J, h):
        """
        Initialize the APT_preprocessor class with J and h.

        :param J: A 2D numpy array representing the coupling matrix (weights J).
        :param h: A 1D numpy array or list representing the external field (biases h).
        """
        self.J = J

        # Convert h to a numpy array if it's a list
        if isinstance(h, list):
            h = np.array(h)

        # If h is a 1D array, reshape it to be a 2D column vector
        if len(h.shape) == 1:
            h = h[:, np.newaxis]

        self.h = h
        self.N = J.shape[0]  # number of spins

    def MCMC(self, num_sweeps, m_start, beta, hash_table=None, use_hash_table=False):
        """
        Implements the Markov Chain Monte Carlo (MCMC) method using Gibbs sampling.

        Parameters:
        - num_sweeps (int): Number of MCMC sweeps to be performed.
        - m_start (np.array[N,]): Initial seed value of the states, where N is the size of the graph.
        - beta (float): Inverse temperature.
        - hash_table (LRUCache, optional): An LRUCache object for storing previously computed dE values.
        - use_hash_table (bool, optional, default=False): If set to True, utilizes the hash table for caching results.

        Returns:
        - M (np.array[N, num_sweeps]): Matrix containing all the sweeps in bipolar form.
        """

        N = self.J.shape[0]
        m = np.asarray(m_start).copy().reshape(-1, 1)
        M = np.zeros((N, num_sweeps))

        for jj in range(num_sweeps):
            spin_state = tuple(m.ravel())

            for kk in np.random.permutation(N):
                if use_hash_table:
                    if not isinstance(hash_table, LRUCache):
                        raise ValueError("hash_table must be an instance of LRUCache")

                    if spin_state in hash_table:
                        dE = hash_table[spin_state]
                    else:
                        dE = self.J.dot(m) + self.h
                        hash_table[spin_state] = dE

                    m[kk] = np.sign(np.tanh(beta * dE[kk]) - 2 * np.random.rand() + 1)
                else:
                    x = self.J.dot(m) + self.h
                    m[kk] = np.sign(np.tanh(beta * x[kk]) - 2 * np.random.rand() + 1)

            M[:, jj] = m.ravel()

        return M

    def MCMC_task(self, m_start, beta, num_sweeps_MCMC, num_sweeps_read, use_hash_table=0):
        """
        Perform a Monte Carlo simulation for a single task.

        This method is designed to be run in a separate parallel process.

        :param m_start: A 1D numpy array representing the initial state.
        :param beta: A float representing the inverse temperature.
        :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps.
        :param num_sweeps_read: An integer representing the number of last sweeps to read.
        :param use_hash_table: A boolean flag. If True, a hash table will be used for caching results. (default = 0)
        :return: A tuple of (Energy, m), where Energy is a 1D numpy array representing the energy at each sweep,
                 and m is the final state after the sweeps.
        """

        # If use_hash_table is True, create a new hash table for this process
        if use_hash_table:
            hash_table = LRUCache(maxsize=10000)
        else:
            hash_table = None

        # Run the MCMC algorithm
        M = self.MCMC(num_sweeps_MCMC, m_start.copy(), beta, hash_table, use_hash_table)

        # Only keep the last num_sweeps_read sweeps
        mm = M[:, -num_sweeps_read:].copy()

        # Initialize an array to store the energy at each sweep
        Energy = np.zeros(num_sweeps_read)

        # Calculate the energy at each sweep
        for kk in range(num_sweeps_read):
            m = mm[:, kk].copy().reshape(1, -1)
            result = -(m @ (self.J / 2) @ m.T + m @ self.h)
            Energy[kk] = result.item()

        # Return the energy and the final state
        return Energy, m

    def run(self, num_sweeps_MCMC=1000, num_sweeps_read=1000, num_rng=100,
            beta_start=0.5, alpha=1.25, sigma_E_val=1000, beta_max=30, use_hash_table=1, num_cores=8):
        """
        Run the Adaptive Parallel Tempering (APT) preprocessing algorithm.

        :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps in each RNG chain (default: 1000).
        :param num_sweeps_read: An integer representing the number of last sweeps to read (default: 1000).
        :param num_rng: An integer representing the number of independent MCMC chains  (default: 100).
        :param beta_start: Initial beta value (default: 0.5). (chosen such that the largest spins can flip often enough)
        :param alpha: alpha value (default: 1.25). (defines the separation of beta in the schedule)
        :param sigma_E_val: initial energy STD value (default: 1000). (set it to large value)
        :param beta_max: Maximum beta value (default: 30). (maximum beta allowed in the schedule)
        :param use_hash_table: Whether to use a hash table or not (default =0).
        :param num_cores: How many CPU cores to use in parallel (default =8).

        """
        foldername = 'data'
        os.makedirs(os.path.join('Results', foldername), exist_ok=True)

        # 1) Normalize energy with |J_ij| ~ 1
        norm_factor = np.max(np.abs(self.J))
        self.J = self.J / norm_factor
        self.h = self.h / norm_factor

        if self.h.shape[0] == 1:
            self.h = self.h.T

        # 2) Initialize β_0 ~ 0.5 (chosen such that the largest spins can flip often enough)
        beta = [deepcopy(beta_start)]
        iter = 1
        sigma_E = [deepcopy(sigma_E_val)]
        sigma_E_min = 0.5 * np.min(np.abs(self.J[self.J != 0]))  # avg_std < σ_min ~ 0.5*(smallest J_ij)
        sigma = []

        saved_state = np.zeros((num_rng, self.N))
        # 4) APT loop until freezeout, typically avg_std < σ_min ~ 0.5*(smallest J_ij)
        while sigma_E > sigma_E_min:
            start_time = time.time()

            if iter != 1:
                # c) Compute new β_i+1 = β_i +(0.85-1.25)/avg_std
                beta.append(beta[-1] + alpha / sigma_E)

            Energy = np.zeros((num_rng, num_sweeps_read))

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = {}
                for j in range(num_rng):
                    if iter == 1:
                        m_start = np.sign(2. * np.random.rand(self.J.shape[0], 1) - 1)
                    else:
                        m_start = saved_state[j, :].copy().reshape(-1, 1)

                    # Submit the task and map it to its index
                    task = executor.submit(self.MCMC_task, m_start.copy(), beta[-1], num_sweeps_MCMC, num_sweeps_read,
                                           use_hash_table)
                    futures[task] = j

                # Process the results in the order they complete
                for future in as_completed(futures):
                    j = futures[future]  # Get the index mapped to this future
                    Energy[j, :], saved_state[j, :] = future.result()

            # b) Compute avg_std = <E_j> over index j
            sigma_E = np.mean(np.std(Energy, axis=1))
            print(f'\ncurrent iteration = {iter}, β = {beta[-1]:.3f}, and average σ = {sigma_E:.3f}\n')

            if beta[-1] > beta_max:
                print('Did not converge but hit the max beta limit\n')
                break

            sigma.append(sigma_E)

            # Save energy and sigma for each iteration
            np.save(os.path.join('Results', foldername, f'Energy_iter_{iter}.npy'), Energy)
            np.save(os.path.join('Results', foldername, f'sigma_iter_{iter}.npy'), sigma_E)

            iter += 1

            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time:.2f} seconds")

        # print('beta =')
        # print(beta)
        # print('sigma =')
        # print(sigma)
        np.save('beta_list_python.npy', beta)
        np.save('sigma_list_python.npy', sigma)
        self.plot_results(beta, sigma)
        return beta, sigma

    def plot_results(self, beta, sigma):
        """
        Plot the beta schedule.
        :param beta: A list of beta values.
        :param sigma: A list of sigma values.
        """
        fig, ax1 = plt.subplots()
        ax1.plot(beta, marker='*', linewidth=2, markersize=6, label='beta')
        ax1.set_ylabel('beta')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.plot(sigma, marker='>', linewidth=2, markersize=6, color='tab:orange', label='sigma')
        ax2.set_ylabel('sigma')
        ax2.tick_params(axis='y')

        ax1.set_xlabel('iteration')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
            label.set_weight('bold')
        #plt.show()
        plt.savefig('beta_sigma.png')


def main():
    # Load J and h from .npy files
    J = np.load('J.npy')
    h = np.load('h.npy')
    J = csr_matrix(J)  # Convert the dense matrix to a sparse one
    # print(J)
    # print(h)

    # Begin preprocessing with APT
    print("\n[INFO] Starting APT preprocessing...")

    # create an APT_preprocessor instance
    apt_prep = APT_preprocessor(J.copy(), h.copy())

    # run Adaptive Parallel Tempering preprocessing
    beta, sigma = apt_prep.run(num_sweeps_MCMC=1000, num_sweeps_read=1000, num_rng=100,
                 beta_start=0.5, alpha=1.25, sigma_E_val=1000, beta_max=64, use_hash_table=0, num_cores=8)

    print("\n[INFO] APT preprocessing complete.")

    beta_list = np.array(beta)
    print(f"[INFO] Beta List: {beta_list}")
    num_replicas = beta_list.shape[0]
    print(f"[INFO] Number of replicas: {num_replicas}")


if __name__ == '__main__':
    main()
