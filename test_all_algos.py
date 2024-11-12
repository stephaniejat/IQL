from framework import value_iteration_episodic, transition_P_RKHS, reward_RKHS, plot_reward_gp_3d, plot_transition_probabilities
from common_framework2 import save_rewards_and_transitions, load_saved_data
import benchmark
import greedy_max_variance  
import without_generative_model
import with_generative_model
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os
import time
import argparse
import pickle


def run_model(model_name, exploration_phase_func, planning_phase_func, M, state_space, action_space, state_action_space, optimal_value_function, beta, exploration_lengths, NUM_RUNS, n_unrolling=20, P_kernel=None, use_D=False):
    save_dir = f"experiment/{model_name}_{P_kernel}_{beta}_{NUM_RUNS}"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    avg_regrets_runs = []
    std_regrets_runs = []
    all_states_list = []
    all_actions_list = []

    for run in range(NUM_RUNS):
        # Initialize wandb
        wandb.init(project="experiment", reinit=True, settings=wandb.Settings(start_method="thread"))
        wandb.run.summary["P_kernel_type"] = P_kernel

        _, _, _, P, r = M
        plot_reward_gp_3d(r, state_space, action_space)
        avg_regrets = []
        std_regrets = []

        all_states = None
        all_actions = None
        D = None

        # Iterate through each exploration length
        for exploration_length in exploration_lengths:
            # Exploration phase
            if use_D:
                if D is None:
                    D, all_states, all_actions, actual_episode = exploration_phase_func(P_kernel, M, exploration_length, state_space,
                                                                                        action_space, state_action_space,
                                                                                        optimal_value_function, beta)
                else:
                    D, all_states, all_actions, actual_episode = exploration_phase_func(P_kernel, M, exploration_length, state_space,
                                                                                        action_space, state_action_space,
                                                                                        optimal_value_function, beta,
                                                                                        D, all_states, all_actions)
            else:
                if all_states is None and all_actions is None:
                    all_states, all_actions, actual_episode = exploration_phase_func(P_kernel, M, exploration_length, state_space,
                                                                                     action_space, state_action_space,
                                                                                     optimal_value_function, beta)
                else:
                    all_states, all_actions, actual_episode = exploration_phase_func(P_kernel, M, exploration_length, state_space,
                                                                                     action_space, state_action_space,
                                                                                     optimal_value_function, beta,
                                                                                     all_states, all_actions)

            # Planning phase
            if use_D:
                avg_regret, std_regret, avg_reward, std_reward = planning_phase_func(P_kernel, M, D,
                                                                                     state_space, action_space,
                                                                                     state_action_space,
                                                                                     optimal_value_function,
                                                                                     beta, n_unrolling)
            else:
                avg_regret, std_regret, avg_reward, std_reward = planning_phase_func(P_kernel, M, all_states,
                                                                                     all_actions, state_space,
                                                                                     action_space, state_action_space,
                                                                                     optimal_value_function,
                                                                                     beta, n_unrolling)

            episode_number = actual_episode + 1
            wandb.log({
                "Model": model_name,
                "Exploration Length": exploration_length,
                "Actual episode": episode_number,
                "Average_Regret": avg_regret,
                "Average_Rewards": avg_reward,
                "Std_Regret": std_regret,
                "Std_Rewards": std_reward,
                "save_dir": save_dir,
                "num_runs": NUM_RUNS
            })

            # Append the average regret and standard deviation to their respective lists at each planning episode
            avg_regrets.append(avg_regret)
            std_regrets.append(std_regret)

        # Store the average regrets and standard deviations for this run 
        avg_regrets_runs.append(avg_regrets)
        std_regrets_runs.append(std_regrets)
        # Save all_states and all_actions for this run
        all_states_list.append(all_states)
        all_actions_list.append(all_actions)

        wandb.log({
            "All_States": all_states_list,
            "All_Actions": all_actions_list,
            "Avg_Regrets_Runs": avg_regrets_runs,
            "Std_Regrets_Runs": std_regrets_runs
        })

    # Save the outputs of the run_model function in npy files inside the specified directory
    np.save(os.path.join(save_dir, 'avg_regrets_runs.npy'), avg_regrets_runs)
    np.save(os.path.join(save_dir, 'std_regrets_runs.npy'), std_regrets_runs)

    with open(os.path.join(save_dir, 'all_states_list.pkl'), 'wb') as f:
        pickle.dump(all_states_list, f)
    with open(os.path.join(save_dir, 'all_actions_list.pkl'), 'wb') as f:
        pickle.dump(all_actions_list, f)

    return avg_regrets_runs, std_regrets_runs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--P_kernel", type=str, default="RBF", help="Kernel type for transition probabilities")
    parser.add_argument("--beta", type=float, default=0.1, help="UCB coefficient")
    parser.add_argument("--num_runs", type=int, default=35, help="Number of runs")
    parser.add_argument("--model_name", type=str, default="Benchmark_Model", help="Name of the model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Regularization for GP regression")
   
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    P_kernel = args.P_kernel
    beta = args.beta
    NUM_RUNS = args.num_runs
    model_name = args.model_name
    alpha = args.alpha

    # Define state and action spaces and their combination
    state_space = np.linspace(0, 1, num=100).reshape(-1, 1)
    action_space = np.linspace(0, 1, num=100).reshape(-1, 1)
    # Define environment parameters
    S = len(state_space)
    A = len(action_space)
    H = 10  # Length of each episode

    # Define save directory
    save_dir = f'Rewards_and_P'
    subdir = os.path.join(save_dir, P_kernel)
    
    if not os.path.exists(subdir) or not os.listdir(subdir):
        save_rewards_and_transitions(P_kernel, alpha, save_dir)
    
    P, r, optimal_value_function = load_saved_data(subdir)
    
    # Continue with other operations using P, r, and optimal_value_function
    print(f"Loaded data from {subdir}")

    # Define state-action space
    state_action_space = np.array([np.hstack((state, action)) for state in state_space for action in action_space])

    # Define environment M = (S, A, H, P, r)
    M = (S, A, H, P, r)

    n_unrolling = 20

    # Define the list of episode lengths for exploration phase
    exploration_lengths = [10, 10, 20, 40, 80]
    
    if model_name == "Benchmark_Model":
        run_model("Benchmark Model",
                  benchmark.exploration_phase,
                  benchmark.planning_phase,
                  M, state_space, action_space, state_action_space,
                  optimal_value_function, beta, exploration_lengths,
                  NUM_RUNS, n_unrolling, P_kernel)
    elif model_name == "Greedy_Max_Variance":
        run_model("Greedy Max Variance",
                  greedy_max_variance.exploration_phase,
                  greedy_max_variance.planning_phase,
                  M, state_space, action_space, state_action_space,
                  optimal_value_function, beta, exploration_lengths,
                  NUM_RUNS, n_unrolling, P_kernel)
    elif model_name == "without_generative_model":
        run_model("Without generative model",
                  without_generative_model.exploration_phase,
                  without_generative_model.planning_phase,
                  M, state_space, action_space, state_action_space,
                  optimal_value_function, beta, [H * term for term in exploration_lengths],
                  NUM_RUNS, n_unrolling, P_kernel, use_D=True)
    elif model_name == "Generative_model":
        run_model("Generative model",
                  with_generative_model.exploration_phase,
                  with_generative_model.planning_phase,
                  M, state_space, action_space, state_action_space,
                  optimal_value_function, beta, exploration_lengths,
                  NUM_RUNS, n_unrolling, P_kernel, use_D=True)
    else:
        # Handle other models here...
        pass


if __name__ == "__main__":
    main()
