from framework import transition_P_RKHS,reward_RKHS,value_iteration_episodic,plot_reward_gp_3d,plot_transition_probabilities
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def plot_transition_probabilities_for_fixed_sprime(P, state_space, action_space, s_prime_idx=9, save_dir=None):
    """
    Plots the transition probabilities P(s'|s,a) for a fixed s' and varying (s, a) pairs.
    
    Args:
    - P: Transition probability matrix of size S x A x S'
    - state_space: List or array of state labels
    - action_space: List or array of action labels
    - s_prime_idx: Index of the fixed s'
    - save_dir: Directory to save the plots (optional)
    """
    # Create labels for states and actions
    state_labels = [str(round(float(state[0]), 2)) for state in state_space]
    action_labels = [str(round(float(action[0]), 2)) for action in action_space]
    
    # Flatten the (s, a) combinations for plotting
    sa_labels = [(s, a) for s in range(len(state_space)) for a in range(len(action_space))]
    probabilities = [P[s, a, s_prime_idx] for s, a in sa_labels]
    
    # Convert (s, a) tuples into readable labels
    #sa_labels_str = [f"s={state_labels[s]}, a={action_labels[a]}" for s, a in sa_labels]
    sa_labels_str = [f"({state_labels[s]}, {action_labels[a]})" for s, a in sa_labels]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.bar(sa_labels_str, probabilities, color='skyblue')
    plt.xlabel("(s, a)", fontsize=25)
    plt.ylabel(r"$P(s'={} \mid s, a)$".format(state_labels[s_prime_idx]), fontsize=25)
    #plt.title(f"Transition Probabilities for Fixed s'={state_labels[s_prime_idx]}", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    
    # Save or display
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"P_fixed_sprime{s_prime_idx}.png")
        plt.savefig(save_path)
        plt.close()

state_space = np.linspace(-1, 1, num=10).reshape(-1, 1)
action_space = np.linspace(-1, 1, num=10).reshape(-1, 1)
P_kernel='RBF'


P=np.load(f'Rewards_and_P_invariant_NEW_0.01/{P_kernel}/P.npy')
r=np.load(f'Rewards_and_P_invariant_NEW_0.01/{P_kernel}/r.npy')
optimal_value_function=np.load(f'Rewards_and_P_invariant_NEW_0.01/{P_kernel}/V.npy')
subdir2='RBF_invariant_0.1'
plot_reward_gp_3d(r,state_space,action_space,subdir2)
plot_transition_probabilities_for_fixed_sprime(P,state_space,action_space,0,subdir2)
plt.plot(np.arange(len(optimal_value_function)), optimal_value_function)
plt.ylabel("optimal value function")
plt.xlabel("state indices")
plt.savefig(os.path.join(subdir2, 'optimal_value_function_plot.png'))

# P_kernel='Matern_smoothness_2.5'


# P=np.load(f'Allrewards/Rewards_and_P_2024-05-08_18-52-52/{P_kernel}/P.npy')
# r=np.load(f'Allrewards/Rewards_and_P_2024-05-08_18-52-52/{P_kernel}/r.npy')
# optimal_value_function=np.load(f'Allrewards/Rewards_and_P_2024-05-08_18-52-52/{P_kernel}/V.npy')

# subdir2='hello_Matern2.5'
# plot_reward_gp_3d(r,state_space,action_space,subdir2)
# plot_transition_probabilities(P,state_space,action_space,subdir2)
# plt.plot(np.arange(len(optimal_value_function)), optimal_value_function)
# plt.ylabel("optimal value function")
# plt.savefig(os.path.join(subdir2, 'optimal_value_function_plot.png'))

# P_kernel='RBF'


# P=np.load(f'Allrewards/Rewards_and_P_2024-05-09_12-48-24/{P_kernel}/P.npy')
# r=np.load(f'Allrewards/Rewards_and_P_2024-05-09_12-48-24/{P_kernel}/r.npy')
# optimal_value_function=np.load(f'Allrewards/Rewards_and_P_2024-05-09_12-48-24/{P_kernel}/V.npy')

# subdir2='hello_RBF'
# plot_reward_gp_3d(r,state_space,action_space,subdir2)
# plot_transition_probabilities(P,state_space,action_space,subdir2)
# plt.plot(np.arange(len(optimal_value_function)), optimal_value_function)
# plt.ylabel("optimal value function")
# plt.savefig(os.path.join(subdir2, 'optimal_value_function_plot.png'))





