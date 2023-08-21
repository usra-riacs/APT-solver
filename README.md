# Readme

The APT-solver is a python package that implements adaptive parallel-tempering (APT). 

This repository contains:

1. A [preprocessing script](https://github.com/usra-riacs/APT-solver/blob/main/apt_preprocessor.py)
2. An [APT script ](https://github.com/usra-riacs/APT-solver/blob/main/apt.py)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Related Efforts](#related-efforts)
- [Contributors](#contributors)
- [License](#license)

## Background

Adaptive Parallel-Tempering is a heuristic method that can be used for optimization and sampling of a target function, for example an Ising model

The goal of this code is to provide a high-performance APT library that can be used in benchmarking emerging hardware, and also used in algorithm design for methods that require an underlying optimization and/or sampling method.

## Installation

### Method 1: Cloning the Repository

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/usra-riacs/APT-solver.git
    cd APT-solver
    ```

2. **Set up a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Method 2: Downloading as a Zip Archive

1. **Download the Repository**:
    - Navigate to the [APT-solver GitHub page](https://github.com/usra-riacs/APT-solver).
    - Click on the `Code` button.
    - Choose `Download ZIP`.
    - Once downloaded, extract the ZIP archive and navigate to the extracted folder in your terminal or command prompt.

2. **Set up a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```


Of course. Here's a concise and formatted `Usage` section in Markdown:

---

I apologize for the oversight. Let's keep it professional and concise. Here's the revised `Usage` section:

---

## Usage

To effectively utilize the APT-solver, follow the outlined steps:

### 1. Setting Up

Ensure you've properly installed the necessary dependencies as highlighted in the [Installation](#installation) section.

### 2. Preprocessing: generate the inverse temperature (beta) schedule

Prepare your data for APT by running the preprocessing code to generate the inverse temperature (beta) schedule and ascertain the number of required replicas:

```python
from apt_preprocessor import APT_preprocessor

# Assuming your data matrices J and h are loaded or generated elsewhere
apt_prep = APT_preprocessor(J, h)

apt_prep.run(num_sweeps_MCMC=1000, num_sweeps_read=1000, num_rng=100,
             beta_start=0.5, alpha=1.25, sigma_E_val=1000, beta_max=64, use_hash_table=0, num_cores=8)
```

### 3. Running Adaptive Parallel Tempering

After preprocessing, proceed with the main Adaptive Parallel Tempering:

```python
from apt import AdaptiveParallelTempering

# Normalize your beta list
beta_list = np.load('beta_list_python.npy')
norm_factor = np.max(np.abs(J))
beta_list = beta_list / norm_factor

apt = AdaptiveParallelTempering(J, h)
M, Energy = apt.run(beta_list, num_replicas=beta_list.shape[0],
                    num_sweeps_MCMC=int(1e4),
                    num_sweeps_read=int(1e3),
                    num_swap_attempts=int(1e2),
                    num_swapping_pairs=1, use_hash_table=0, num_cores=8)
```

For more detailed examples, refer to the [examples](https://github.com/usra-riacs/APT-solver/tree/main/examples) directory in the repository.

---

Again, I apologize for the earlier oversight. Thank you for pointing it out.
### 4. Example Script

For a full demonstration of the APT-solver in action, refer to the example script located in the `examples` folder of this repository.

---


## Related Efforts
- [PySA](https://github.com/nasa/PySA) - High Performance NASA QuAIL Python Simulated Annealing code

## Contributors
- [@NavidAaddit](https://github.com/navidaadit) Navid Aaddit
- [@PAaronLott](https://github.com/PAaronLott) Aaron Lott
- [@Mohseni](https://github.com/mohseni7) Masoud Mohseni

## Acknowledgements

This code was developed under the NSF Expeditions Program NSF award CCF-1918549 on [Coherent Ising Machines](https://cohesing.org/)

## License

[Apache2](LICENSE)
