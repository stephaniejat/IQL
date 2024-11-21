import test_kernelized_RL
from test_kernelized_RL import transition_P, value_iteration_episodic, reward_function,transition_dynamics
import numpy as np
import wandb

def main():
    #Define state and action spaces and their combination
    state_space = np.linspace(0, 1, num=100).reshape(-1, 1)
    action_space = np.linspace(0, 1, num=100).reshape(-1, 1)

    P=transition_P(state_space,action_space)
    print('P',P)
    optimal_value_function = value_iteration_episodic(state_space,action_space,reward_function,P,10)
    print('optimal_value_function',optimal_value_function)
     #Define state and action spaces and their combination
    state_action_space = []
    for state in state_space:
        for action in action_space:
            state_action_space.append(np.hstack((state, action)))
    state_action_space = np.array(state_action_space)

        # Define environment parameters
    S = len(state_space)
    A = len(action_space)
    H = 10  # Length of each episode
    r = reward_function  # Not used in this example

    # Define environment M = (S,A,H,P,r)
    M = (S, A, H, P, r)

    # Run experiments with p-KRVI policy
    T = 1000 # Number of episodes
    beta=0.1 # UCB coefficient
    NUM_RUNS=1
    #wandb.init(project="kernelized_RL")


    # pi_krvi_policy(M, T,state_space, action_space,optimal_value_function,beta)

    #,name=f"run_{run}"
    for run in range(NUM_RUNS):
        # Start a new run for each iteration
        wandb.init(project="test", reinit=True)
        test_kernelized_RL.pi_krvi_policy(M, T, state_space, action_space,state_action_space,optimal_value_function, beta)
        # Log metrics specific to this run
        wandb.log({"run": run})

if __name__ == "__main__":
    main()

