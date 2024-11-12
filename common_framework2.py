from framework import transition_P_RKHS, reward_RKHS, value_iteration_episodic, plot_reward_gp_3d, plot_transition_probabilities
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

def save_rewards_and_transitions(P_kernel, alpha, save_dir):
    state_space = np.linspace(0, 1, num=100).reshape(-1, 1)
    action_space = np.linspace(0, 1, num=100).reshape(-1, 1)
    H = 10
    
    # Create subdirectory for P_kernel if it doesn't exist
    subdir = os.path.join(save_dir, P_kernel)
    os.makedirs(subdir, exist_ok=True)
    
    P = transition_P_RKHS(state_space, action_space, P_kernel, alpha)
    r = reward_RKHS(P_kernel, state_space, action_space, subdir, alpha)
    optimal_value_function = value_iteration_episodic(state_space, action_space, r, P, H)
    
    # Save P, r, and optimal_value_function
    np.save(os.path.join(subdir, 'P.npy'), P)
    np.save(os.path.join(subdir, 'r.npy'), r)
    np.save(os.path.join(subdir, 'V.npy'), optimal_value_function)
    plot_reward_gp_3d(r, state_space, action_space, save_dir)
    # plot_transition_probabilities(P, state_space, action_space, subdir)
    # plt.plot(np.arange(len(optimal_value_function)), optimal_value_function)
    # plt.ylabel("Optimal Value Function")
    # plt.savefig(os.path.join(subdir, 'optimal_value_function_plot.png'))
    # plt.close()

def load_saved_data(subdir):
    P = np.load(os.path.join(subdir, 'P.npy'))
    r = np.load(os.path.join(subdir, 'r.npy'))
    optimal_value_function = np.load(os.path.join(subdir, 'V.npy'))
    return P, r, optimal_value_function

