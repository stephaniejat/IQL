# IQL
Kernelized Q-learning with invariances.

# KRVI Algorithm for Rotated Wrapped Frozen Lake

## Overview

This code implements the **KRVI**  algorithm for testing on the **rotated wrapped frozen lake** environment. The core of the experiment involves selecting from the original **4x4 Frozen Lake** environment or one of its three rotated configurations (by 90°, 180°, and 270° clockwise) uniformly at random in each episode. The algorithm leverages a **Gaussian Process (GP)** with a kernel that is either the **Invariant Kernel** or a standard **RBF Kernel**.

This version of the algorithm includes options for hyperparameter optimization using **botorch**, logging with **WandB**, and data logging in a CSV format.
# Functionality

### Rotated Frozen Lake
Each episode selects a random rotation of the environment. The agent can play the original 4x4 Frozen Lake environment or one of its three rotated configurations: by 90°, 180°, or 270° clockwise.

### Invariant Kernel
The core idea behind KRVI is the use of a kernel-based approach for value iteration, utilizing an Invariant Kernel for Gaussian Process regression to handle symmetries in the environment. Alternatively, the standard RBF Kernel can be used by changing the kernel in the code.

## Training Process:
- The agent interacts with the environment for a specified number of episodes (iterations).
- The value function is approximated using Gaussian Processes.
- The agent makes decisions based on predicted Q-values for each action.
- Q-values are predicted using the kernelized GP regression at each timestep and are used to select actions.

## Logging:
- **CSV File**: Training metrics such as rewards, cumulative returns, and episode number are saved to a CSV file (`krvi_metrics.csv`).
- **WandB Logging**: Optionally, the training process can be logged to WandB for visualization and analysis. Logging is disabled by default by setting `WANDB_DISABLED = true`.

# Key Classes and Functions

### KRVI
The `KRVI` class implements the core logic of the algorithm. It handles the interaction with the environment, training the Gaussian Process (GP), and logging results.

#### Key Parameters:
- **kernel**: A callable kernel function (either InvariantKernel or RBFKernel).
- **env**: The environment to train on (e.g., `"FrozenLake-v1"`).
- **beta**: UCB coefficient for exploration vs. exploitation trade-off.
- **horizon**: Number of timesteps per episode.
- **action_transformation**: A callable that maps actions from the environment's action space.
- **len_scale**: Length scale for the kernel.
- **noise_reg**: Noise regularization for the Gaussian Process.
- **optim_botorch**: If set to 1, hyperparameter optimization is enabled using botorch.

#### Main Methods:
- **train**: The main training loop of the KRVI algorithm. This method runs through the specified number of episodes (T), performs kernelized regression at each timestep, and logs results.
- **GP_regression_torch**: Performs Gaussian Process regression using the specified kernel, optimizing the GP model parameters using botorch if necessary.
- **predict_with_gp**: Makes predictions using the trained Gaussian Process, returning the predicted Q-values, means, and standard deviations.

### train
The main training loop of the KRVI algorithm. This method performs the following actions:
- Selects a random rotation of the environment.
- Trains the GP on the state-action pairs observed in previous episodes.
- Selects actions based on the Q-values predicted by the GP model.

### GP_regression_torch
Performs Gaussian Process regression using the specified kernel, optimizing the GP model parameters using botorch if necessary.

### predict_with_gp
Makes predictions using the trained Gaussian Process, returning the predicted Q-values, means, and standard deviations.

# How to Run

1. Clone the repository or save the code to a `.py` file.
2. Ensure that you have the required dependencies installed.
3. Adjust the configuration options or hyperparameters as needed, using command-line arguments.

### Example Command
To run the training for 2000 episodes with custom hyperparameters:

```bash
python KRVI_algo_test_rotated_modified.py --beta 0.1 --horizon 100 --len_scale 0.1 --noise_reg 0.1 --env "FrozenLake-v1" --logging "trial" --iterations 2000 --verbose 1 --optim_botorch 1
```


### Customization

## Using the RBF Kernel

If you prefer using the standard RBF kernel instead of the invariant kernel, you can change the kernel line when initializing KRVI:

```python
k_G = RBFKernel()  # Uncomment to use RBF Kernel



