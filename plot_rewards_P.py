from framework import transition_P_RKHS,reward_RKHS,value_iteration_episodic,plot_reward_gp_3d,plot_transition_probabilities
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

state_space = np.linspace(0, 1, num=100).reshape(-1, 1)
action_space = np.linspace(0, 1, num=100).reshape(-1, 1)
P_kernel='Matern_smoothness_1.5'


P=np.load(f'Rewards_and_P_paper/Rewards_and_P_2024-05-08_17-03-10/{P_kernel}/P.npy')
r=np.load(f'Rewards_and_P_paper/Rewards_and_P_2024-05-08_17-03-10/{P_kernel}/r.npy')
optimal_value_function=np.load(f'Rewards_and_P_paper/Rewards_and_P_2024-05-08_17-03-10/{P_kernel}/V.npy')
subdir2='hello_Matern1.5_2'
plot_reward_gp_3d(r,state_space,action_space,subdir2)
plot_transition_probabilities(P,state_space,action_space,subdir2)
plt.plot(np.arange(len(optimal_value_function)), optimal_value_function)
plt.ylabel("optimal value function")
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





