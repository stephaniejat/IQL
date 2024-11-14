from framework import transition_P_RKHS, reward_RKHS, value_iteration_episodic, plot_reward_gp_3d, plot_transition_probabilities
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def main():
    state_space = np.linspace(0, 1, num=10).reshape(-1, 1)
    action_space = np.linspace(0, 1, num=10).reshape(-1, 1)
    P_kernel = "RBF"
    #grid_size = 10 # Grid size for fitting GP regression
    H=10

    # Generate all possible input points in the grid (state-action pairs)
    # values = np.linspace(0, 1, grid_size)
    # X = np.array(list(product(values, repeat=2)))  # 2D grid points for state-action pairs

    # state_transformations = [lambda x: x]  # Identity transformation for state space
    # action_transformations = [lambda x: x]  # Identity transformation for action space

    # Loop through each transformation combination (currently just identity functions)
    # for g1, g2 in product(state_transformations, action_transformations):
    #     for x in X:
    #         # Apply the transformations to state and action
    #         transformed_state = g1(x[0])
    #         transformed_action = g2(x[1])

    #         # Check and print the shapes of the transformed state and action
    #         print(f"Shape of transformed state: {transformed_state}, Shape of transformed action: {transformed_action}")

    #         # Combine state and action into one array (ensure correct shape)
    #         X_transformed = np.array([transformed_state, transformed_action])
    #         print(f"Shape of X_transformed: {X_transformed}")

    # Example of how to call the reward function, if uncommented and available
    r = reward_RKHS(P_kernel, state_space, action_space, 'Hello', alpha=0.5)
    plot_reward_gp_3d(r, state_space, action_space, 'Hello')
    P = transition_P_RKHS(state_space, action_space, P_kernel, alpha=0.5)
    #plot_transition_probabilities(P, state_space, action_space, 'Hello')
    #optimal_value_function = value_iteration_episodic(state_space, action_space, r, P, H)

if __name__ == "__main__":
    main()
