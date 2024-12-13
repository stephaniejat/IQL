from framework import transition_P_RKHS, reward_RKHS, value_iteration_episodic, plot_reward_gp_3d, plot_transition_probabilities
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def main():
    state_space = np.linspace(-1, 1, num=100).reshape(-1, 1)
    action_space = np.linspace(-1, 1, num=100).reshape(-1, 1)
    P_kernel = "RBF"
    #grid_size = 10 # Grid size for fitting GP regression
    H=10
    # Example of how to call the reward function, if uncommented and available
    r = reward_RKHS(P_kernel, state_space, action_space, 'Hello', alpha=0.05)
    plot_reward_gp_3d(r, state_space, action_space, 'Hello')
    P = transition_P_RKHS(state_space, action_space, P_kernel, alpha=0.5)
    plot_transition_probabilities(P, state_space, action_space, 'Hello')
    optimal_value_function = value_iteration_episodic(state_space, action_space, r, P, H)
    print('opt V', optimal_value_function)
    state_indices = np.arange(len(optimal_value_function))  # State indices (0 to n-1)
    plt.figure(figsize=(8, 6))
    plt.plot(state_indices, optimal_value_function, marker='o', label='Value Function')
    plt.title('Value Function Across States')
    plt.xlabel('State Index')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.plot(optimal_value_function)
    plt.savefig('Optimal_V')

if __name__ == "__main__":
    main()
