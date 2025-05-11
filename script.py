import test_kernelized_RL_invariant
#from framework import value_iteration_episodic, transition_P_RKHS, reward_RKHS, plot_reward_gp_3d, plot_transition_probabilities, InvariantKernel
from common_framework2 import save_rewards_and_transitions, load_saved_data
import numpy as np
import wandb
import os
from datetime import datetime

def main():
    #Define state and action spaces and their combination
    state_space = np.linspace(-1, 1, num=10).reshape(-1, 1)
    action_space = np.linspace(-1, 1, num=10).reshape(-1, 1)

  # Define state-action space
    state_action_space = np.array([np.hstack((state, action)) for state in state_space for action in action_space])
        # Define environment parameters
    S = len(state_space)
    A = len(action_space)
    H = 10  # Length of each episode

    P_kernel= 'RBF'
    alpha=0.01 #0.001
     # Define save directory
    #save_dir = f'Rewards_and_P_invariant_NEW'
    #subdir = os.path.join(save_dir, P_kernel)
    # Add a timestamp to the save directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Example format: 2024-11-19_15-30-45
    save_dir = f'Rewards_and_P_invariant_NEW_{alpha}'

    subdir = os.path.join(save_dir, P_kernel)
    
    if not os.path.exists(subdir) or not os.listdir(subdir):
        print('not there')
        save_rewards_and_transitions(P_kernel, alpha, save_dir)
    
    P, r, optimal_value_function = load_saved_data(subdir)
    # print('P',P)
    # print('P.shape',P.shape)
    # print('r',r)
    # print('r.shape',r.shape)
    # print('optimal_value_function',optimal_value_function)
    # print('optimal_value_function_shape',optimal_value_function.shape)

    

    # Define environment M = (S,A,H,P,r)
    M = (S, A, H, P, r)

    # Run experiments with p-KRVI policy
    T = 1000 # Number of episodes 
    beta=0.1 # UCB coefficient (0.001)
    NUM_RUNS=20 
    #wandb.init(project="kernelized_RL")


    # pi_krvi_policy(M, T,state_space, action_space,optimal_value_function,beta)

    #,name=f"run_{run}"
    for run in range(NUM_RUNS):
        # Start a new run for each iteration
        wandb.init(project="IQL_submission_test", reinit=True)
        wandb.run.summary["alpha_functions"] = alpha
        test_kernelized_RL_invariant.pi_krvi_policy(M, T, state_space, action_space,state_action_space,optimal_value_function, beta)
        # Log metrics specific to this run
        wandb.log({"run": run})

if __name__ == "__main__":
    main()

