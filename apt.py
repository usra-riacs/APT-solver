import random
import time
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from cachetools import LRUCache
from random import randint


# np.random.seed(12624755)  # Set the seed to an arbitrary number

class AdaptiveParallelTempering:
    """
    The AdaptiveParallelTempering class is used to implement the Adaptive Parallel Tempering algorithm.
    """

    def __init__(self, J, h, num_replicas, num_sweeps_MCMC, num_sweeps_read, num_swap_attempts, num_swapping_pairs=1):
        """
        Initialize an AdaptiveParallelTempering object.

        :param num_replicas: An integer, the number of replicas (parallel chains) to use in the algorithm.
        :param num_sweeps_MCMC: An integer, the number of Monte Carlo sweeps to perform.
        :param num_sweeps_read: An integer, the number of last sweeps to read from the chains.
        :param num_swap_attempts: An integer, the number of swap attempts between chains.
        :param num_swapping_pairs: An integer, the number of replica pairs per swap attempt (default =1)
        :param J: A 2D numpy array representing the interaction matrix in a spin glass model.
        :param h: A 1D numpy array representing the external magnetic field in a spin glass model.
        """
        self.J = J
        self.h = h
        self.num_replicas = num_replicas
        self.num_sweeps_MCMC = num_sweeps_MCMC
        self.num_sweeps_read = num_sweeps_read
        self.num_swap_attempts = num_swap_attempts
        self.num_sweeps_MCMC_per_swap = self.num_sweeps_MCMC // self.num_swap_attempts
        self.num_sweeps_read_per_swap = self.num_sweeps_read // self.num_swap_attempts
        self.num_swapping_pairs = num_swapping_pairs
        self.colorMap = self.greedy_coloring_saturation_largest_first()

    def replica_energy(self, M, num_sweeps):
        """
        Calculate the energy of a given replica over a number of sweeps.

        :param M: A 2D numpy array representing the MCMC state after each sweep.
        :param num_sweeps: An integer representing the number of sweeps.

        :return: A tuple where the first element is the minimum energy and the second element is an array of energies.
        """
        EE1 = np.zeros(num_sweeps)
        for ii in range(num_sweeps):
            m1 = M[:, ii]
            EE1[ii] = -1 * (m1.T @ self.J @ m1 / 2 + m1.T @ self.h)
        minEnergy = np.min(EE1)
        return minEnergy, EE1

    def greedy_coloring_saturation_largest_first(self):
        """
        Perform greedy coloring using the saturation largest first strategy.

        :return colorMap: A 1D numpy array containing the colormap of the graph.
        """
        # Create a NetworkX graph from the J matrix
        G = nx.Graph(self.J)

        # Perform greedy coloring with the saturation largest first strategy
        color_map = nx.coloring.greedy_color(G, strategy='saturation_largest_first')

        # Convert the color map to a 1D numpy array
        colorMap = np.array([color_map[node] for node in G.nodes])

        return colorMap

    def MCMC_GC(self, num_sweeps_MCMC, m_start, beta, hash_table, use_hash_table=0):
        """
        Perform MCMC with graph coloring.

        :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps.
        :param m_start: A 1D numpy array representing the initial state.
        :param beta: A float representing the inverse temperature.
        :param hash_table: A LRUCache object used to store previously calculated dE values.
        :param use_hash_table: A boolean flag. If True, a hash table will be used for caching results. (default = 0)

        :return M: A 2D numpy array representing the MCMC state after each sweep.
        """
        N = self.J.shape[0]
        m = m_start
        M = np.zeros((N, num_sweeps_MCMC))
        J = csr_matrix(self.J)

        if self.h.shape[0] == 1:
            self.h = self.h.T

        # Group spins by color
        required_colors = len(np.unique(self.colorMap))
        Groups = [None] * required_colors
        for k in range(required_colors):
            Groups[k] = np.where(self.colorMap == k)[0]

        # Create a list of grouped J and h matrices
        J_grouped = [J[Groups[k], :] for k in range(required_colors)]
        h_grouped = [self.h[Groups[k]] for k in range(required_colors)]

        for jj in range(num_sweeps_MCMC):
            for ijk in range(required_colors):
                group = Groups[ijk]
                spin_state = tuple(m.ravel())

                if use_hash_table:
                    if not isinstance(hash_table, LRUCache):
                        raise ValueError("hash_table must be an instance of cachetools.LRUCache")

                    if spin_state in hash_table:
                        dE = hash_table[spin_state]
                    else:
                        dE = J.dot(m) + self.h
                        hash_table[spin_state] = dE

                    m[group] = np.sign(np.tanh(beta * dE[group]) - 2 * np.random.rand(len(group), 1) + 1)
                else:
                    x = J_grouped[ijk].dot(m) + h_grouped[ijk]
                    m[group] = np.sign(np.tanh(beta * x) - 2 * np.random.rand(len(group), 1) + 1)

            M[:, jj] = m.ravel()

        return M

    def MCMC_task(self, replica_i, num_sweeps_MCMC, m_start, beta_list, use_hash_table=0):
        """
        Perform a Monte Carlo simulation for a single task.

        This method is designed to be run in a separate process.

        :param m_start: A 1D numpy array representing the initial state.
        :param beta_list: A 1D numpy array representing the inverse temperatures for the replicas.
        :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps.
        :param use_hash_table: A boolean flag. If True, a hash table will be used for caching results. (default = 0)
        """

        # If use_hash_table is True, create a new hash table for this process
        if use_hash_table:
            hash_table = LRUCache(maxsize=10000)
        else:
            hash_table = None
        return self.MCMC_GC(num_sweeps_MCMC, m_start, beta_list[replica_i - 1], hash_table, use_hash_table)

    def select_non_overlapping_pairs(self, all_pairs):
        """
        Select non-overlapping pairs from a list of all possible consecutive pairs.

        :param all_pairs: A list of tuples where each tuple is a consecutive pair.

        :return: A list of non-overlapping pairs (a total of num_swapping_pairs).
        """
        available_pairs = all_pairs.copy()
        selected_pairs = []
        for _ in range(self.num_swapping_pairs):
            if not available_pairs:
                raise ValueError("Cannot find non-overlapping pairs.")
            i_pair = randint(0, len(available_pairs) - 1)
            pair = available_pairs[i_pair]
            selected_pairs.append(pair)
            # Remove pairs that overlap with the chosen pair
            available_pairs = [p for p in available_pairs if
                               p[0] != pair[0] and p[0] != pair[1] and p[1] != pair[0] and p[1] != pair[1]]
        return selected_pairs

    def run(self, beta_list):
        """
        Run the adaptive parallel tempering algorithm.
        :param beta_list: A 1D numpy array representing the inverse temperatures for the replicas.
        """
        num_spins = self.J.shape[0]
        count = np.zeros(self.num_swap_attempts)
        swap_attempted_replicas = np.zeros((self.num_swap_attempts * self.num_swapping_pairs, 2))
        swap_accepted_replicas = np.zeros((self.num_swap_attempts * self.num_swapping_pairs, 2))

        # Generate all possible consecutive pairs of replicas
        all_pairs = [(i, i + 1) for i in range(1, self.num_replicas)]

        # Initialize states for all replicas
        M = np.zeros((self.num_replicas * num_spins, self.num_sweeps_MCMC))
        m_start = np.sign(2 * np.random.rand(self.num_replicas * num_spins, 1) - 1)
        use_hash_table = 0

        swap_index = 0

        with ProcessPoolExecutor(max_workers=8) as executor:
            for ii in range(self.num_swap_attempts):
                print(f"\nRunning swap attempt = {ii + 1}")
                start_time = time.time()

                # Run MCMC for each replica in parallel
                futures = [executor.submit(self.MCMC_task, replica_i, self.num_sweeps_MCMC,
                                           m_start[(replica_i - 1) * num_spins:replica_i * num_spins], beta_list,
                                           use_hash_table) for replica_i in range(1, self.num_replicas + 1)]
                M_results = [future.result() for future in futures]

                for replica_i, M_replica in enumerate(M_results, start=1):
                    M[(replica_i - 1) * num_spins:replica_i * num_spins, :] = M_replica

                # Truncate the state matrix for reading and update the starting states
                mm = M[:, -self.num_sweeps_read_per_swap:].T
                m_start = M[:, -1].reshape(-1, 1)

                selected_pairs = self.select_non_overlapping_pairs(all_pairs)

                # Attempt to swap states of each selected pair of replicas
                for pair in selected_pairs:
                    sel, next = pair
                    m_sel = mm[-1, (sel - 1) * num_spins:sel * num_spins].T
                    m_next = mm[-1, (next - 1) * num_spins:next * num_spins].T

                    E_sel = -m_sel.T @ self.J @ m_sel / 2 - m_sel.T @ self.h
                    E_next = -m_next.T @ self.J @ m_next / 2 - m_next.T @ self.h
                    beta_sel = beta_list[sel - 1]
                    beta_next = beta_list[next - 1]

                    print(f"\nSelected pair indices: {sel}, {next}")
                    print(f"β values: {beta_sel}, {beta_next}")
                    print(f"Energies: {E_sel}, {E_next}")

                    swap_attempted_replicas[swap_index, :] = [sel, next]

                    DeltaE = E_next - E_sel
                    DeltaB = beta_next - beta_sel

                    if np.random.rand() < min(1, np.exp(DeltaB * DeltaE)):
                        count[ii] += 1
                        swap_accepted_replicas[swap_index, :] = [sel, next]
                        print(f"Swapping {int(sum(count))}th time")

                        # Swap the states of the selected replicas
                        m_start[(sel - 1) * num_spins:sel * num_spins] = m_next.reshape(-1, 1)
                        m_start[(next - 1) * num_spins:next * num_spins] = m_sel.reshape(-1, 1)

                    swap_index += 1

                elapsed_time = time.time() - start_time
                print(f"Elapsed time for swap attempt {ii + 1}: {elapsed_time}")

        # Calculate the final energies of the replicas
        Energy = np.zeros(self.num_replicas)
        EE1_list = []
        for look_replica in range(1, self.num_replicas + 1):
            M_replica = M[(look_replica - 1) * self.J.shape[1]:look_replica * self.J.shape[1], :]
            minEnergy, EE1 = self.replica_energy(M_replica, self.num_sweeps_read)
            Energy[look_replica - 1] = minEnergy
            EE1_list.append(EE1)

        # Output the results
        print(f"\nLatest energy from each replica = {Energy}")
        print(f"Swap acceptance rate = {np.count_nonzero(count) / count.size * 100:.2f} per cent\n")

        # Plot the energy traces
        self.plot_energies(EE1_list, beta_list)

    def plot_energies(self, EE1_list, beta_list):
        """
        Plot the energy traces of all replicas.

        :param EE1_list: A list of 1D numpy arrays representing the energy traces of the replicas.
        :param beta_list: A 1D numpy array representing the inverse temperatures for the replicas.
        """
        plt.figure()
        for i in range(self.num_replicas):
            plt.plot(EE1_list[i], label=f"Replica {i + 1} (β={beta_list[i]:.2f})")
        plt.xlabel('Sweeps')
        plt.ylabel('Energy')
        plt.title('Energy traces for different replicas')
        plt.legend()
        plt.show()
        # plt.savefig('APT_energy.png')


def main():

    # Load the coupling and external field matrices, and the list of inverse temperatures
    J = np.load('J.npy')
    h = np.load('h.npy')
    J = csr_matrix(J)  # Convert the dense matrix to a sparse one
    beta_list = np.load('beta_list_python.npy')
    num_replicas = beta_list.shape[0]

    # uncomment if you want to manually select the inverse temperatures for the replicas from beta_list
    # startingBeta = 18
    # num_replicas = 12
    # selectedBeta = range(startingBeta, startingBeta + num_replicas)
    # beta_list = beta_list[selectedBeta]
    norm_factor = np.max(np.abs(J))
    beta_list = beta_list / norm_factor
    print(beta_list)

    # Create an AdaptiveParallelTempering instance and run it
    ap = AdaptiveParallelTempering(
        J=J,
        h=h,
        num_replicas=num_replicas,
        num_sweeps_MCMC=int(1e4),
        num_sweeps_read=int(1e3),
        num_swap_attempts=int(1e2),
        num_swapping_pairs=3
    )
    ap.run(beta_list)


if __name__ == '__main__':
    main()
