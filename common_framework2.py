from framework import transition_P_RKHS, reward_RKHS, value_iteration_episodic, plot_reward_gp_3d, plot_transition_probabilities
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

def save_rewards_and_transitions(P_kernel, alpha, save_dir):
    state_space = np.linspace(-1, 1, num=10).reshape(-1, 1)
    action_space = np.linspace(-1, 1, num=10).reshape(-1, 1)
    H = 10
    
    subdir = os.path.join(save_dir, P_kernel)
    os.makedirs(subdir, exist_ok=True)
    
    P = transition_P_RKHS(state_space, action_space, P_kernel, alpha)
    r = reward_RKHS(P_kernel, state_space, action_space, subdir, alpha)
    optimal_value_function = value_iteration_episodic(state_space, action_space, r, P, H)
    
    np.save(os.path.join(subdir, 'P.npy'), P)
    np.save(os.path.join(subdir, 'r.npy'), r)
    np.save(os.path.join(subdir, 'V.npy'), optimal_value_function)
    plot_reward_gp_3d(r, state_space, action_space, save_dir)
    plot_transition_probabilities(P, state_space,action_space, save_dir)
    plt.plot(optimal_value_function)
    plt.ylabel("optimal value function")
    save_path = os.path.join(save_dir, f"optimal_V.png")
    plt.savefig(save_path)
   


def load_saved_data(subdir):
    P = np.load(os.path.join(subdir, 'P.npy'))
    r = np.load(os.path.join(subdir, 'r.npy'))
    optimal_value_function = np.load(os.path.join(subdir, 'V.npy'))
    return P, r, optimal_value_function

