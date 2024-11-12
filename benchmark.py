import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern
from scipy.stats import truncnorm
import wandb
from framework import transition_dynamics


def GP_regression(X, y, state_action_space,P_kernel,l=0.1,alpha=0.5):
    tau=alpha
    if P_kernel == "Matern_smoothness_1.5":
        kernel = Matern(length_scale=l, nu=1.5, length_scale_bounds="fixed")
    elif P_kernel == "Matern_smoothness_2.5":
        kernel = Matern(length_scale=l, nu=2.5, length_scale_bounds="fixed")
    elif P_kernel =="RBF":
        kernel = RBF(length_scale=l,length_scale_bounds="fixed")
        tau= 0.01
 

    wandb.run.summary["kernel type"] = kernel
    wandb.run.summary["alpha"] = tau
    wandb.run.summary["length_scale"] = l
    # Create GPR model
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None,alpha=tau) 
    # Fit the model
    gpr.fit(X, y)
    # Predict mean and standard deviation
    y_pred_mean, y_pred_std = gpr.predict(state_action_space, return_std=True)
    
    return y_pred_mean, y_pred_std




def exploration_phase(P_kernel,M, T, state_space, action_space,state_action_space, optimal_V,beta,all_states=None,all_actions=None):
    S, A, H, P, r = M
    
    wandb.run.summary["episode length"] = H
    wandb.run.summary["episode number"] = T
    wandb.run.summary["UCB coef"] = beta

    # These arrays store the observations and actions over all episodes
    if all_states is None:
        all_states=[]
        all_actions=[]
    
    # Get the starting episode number
    start_episode = 0
    if all_states:
        start_episode = len(all_states)


    ft_mean = np.zeros((H+1, len(state_action_space)))
    ft_std = np.zeros((H+1, len(state_action_space)))
    ft_estimate = np.zeros((H+1, len(state_space), len(action_space)))
    ucb_bonus = np.zeros((H+1, len(state_space), len(action_space)))
    exploration_reward = np.zeros((H+1, len(state_space), len(action_space)))
    Qt_estimate = np.zeros((H+1, len(state_space), len(action_space)))
 
    for episode in range(T): #T here are the new episodes only
        actual_episode=start_episode + episode
  
        # Collecting (s,a) pairs and (V(s')) from previous episodes
        if actual_episode>0:
            for h in reversed(range(H)):
                X_states = np.concatenate([all_states[i][h] for i in range(actual_episode)])
                X_actions = np.concatenate([all_actions[i][h] for i in range(actual_episode)])
        
                # Reshape X_states and X_actions to be 2D arrays
                X_states = X_states.reshape(-1, 1)
                X_actions = X_actions.reshape(-1, 1)
        
                # Concatenate X_states and X_actions along axis 1
                X = np.concatenate((X_states, X_actions), axis=1)
                
                Vnext = []
                for i in range(actual_episode):
                    if h < H - 1:
                        next_state_index = np.argmin(np.abs(state_space - all_states[i][h + 1]))
                        Vnext.append(np.max(Qt_estimate[h + 1][next_state_index, :]))
                    else:
                        # Handle the case where h is at the last step of the episode
                        Vnext.append(0)  

                y = np.array([Vnext[i] for i in range(len(Vnext))])
                ft_mean[h], ft_std[h] = GP_regression(X, y, state_action_space,P_kernel)
                ft_mean_reshaped = ft_mean[h].reshape((len(state_space), len(action_space)))
                ft_std_reshaped = ft_std[h].reshape((len(state_space), len(action_space)))
                ucb_bonus[h]=np.minimum(beta * ft_std_reshaped,H)
                exploration_reward[h]= ucb_bonus[h] / H
                ft_estimate[h] = ft_mean_reshaped 
                q= ft_estimate[h] + exploration_reward[h] + ucb_bonus[h]
                Qt_estimate[h]= np.minimum(np.maximum(q, 0), H)
    
          
        # Initialize arrays to store observations for the current episode
        episode_states = []
        episode_actions = []
        episode_rewards = []

        initial_state_index = 20 # Fix initial state
        state = state_space[initial_state_index]  
    
        # Loop to choose actions greedily and collect observations
        for h in range(H):
         
            state_index = np.argmin(np.abs(state_space - state))  # Find index of the current state
            # Choose action greedily based on Q-values
            q_values=Qt_estimate[h][state_index]  # Get optimistic Q-values for current state
            action_index = np.argmax(q_values)  # Find index of action with highest Q-value (greedy policy)
            action = action_space[action_index]  # Choose action with highest Q-value
            # Proceed with the chosen action
            next_state = transition_dynamics(state, action,P,state_space,action_space)
            reward=r[state_index,action_index]

            # Store observations
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state 
        
      
      
        all_states.append(np.array(episode_states))
        all_actions.append(np.array(episode_actions))
       

    return all_states,all_actions,actual_episode

########### planning phase ########
def planning_phase(P_kernel,M, all_states,all_actions, state_space, action_space,state_action_space, optimal_V,beta,n_unrolling=20):
    S, A, H, P, r = M

    ft_mean = np.zeros((H+1, len(state_action_space)))
    ft_std = np.zeros((H+1, len(state_action_space)))
    ft_estimate = np.zeros((H+1, len(state_space), len(action_space)))

    ucb_bonus = np.zeros((H+1, len(state_space), len(action_space)))
    Qt_estimate = np.zeros((H+1, len(state_space), len(action_space)))
    reward = np.zeros((H, len(state_space), len(action_space)))
   
    for h in reversed(range(H)):
        X_states = np.concatenate([all_states[i][h] for i in range(len(all_states))])
        X_actions = np.concatenate([all_actions[i][h] for i in range(len(all_states))])

        # Reshape X_states and X_actions to be 2D arrays
        X_states = X_states.reshape(-1, 1)
        X_actions = X_actions.reshape(-1, 1)

        # Concatenate X_states and X_actions along axis 1
        X = np.concatenate((X_states, X_actions), axis=1)
        
        Vnext = []
        for i in range(len(all_states)):
            if h < H - 1:
                next_state_index = np.argmin(np.abs(state_space - all_states[i][h + 1]))
                Vnext.append(np.max(Qt_estimate[h + 1][next_state_index, :]))
            else:
                # Handle the case where h is at the last step of the episode
                Vnext.append(0) 

        y = np.array([Vnext[i] for i in range(len(Vnext))])
        ft_mean[h], ft_std[h] = GP_regression(X, y, state_action_space,P_kernel)
        ft_mean_reshaped = ft_mean[h].reshape((len(state_space), len(action_space)))
        ft_std_reshaped = ft_std[h].reshape((len(state_space), len(action_space)))

        # Calculate Qt_estimate[h] from Qt_mean and Qt_std
        ucb_bonus[h]=np.minimum(beta * ft_std_reshaped,H)
        ft_estimate[h] = ft_mean_reshaped 
        temp_reward = np.zeros_like(ft_mean_reshaped)  # Initialize reward array

        for i, state in enumerate(state_space):
            for j, action in enumerate(action_space):
                temp_reward[i, j] = r[i,j]
        reward[h]=temp_reward
        q= ft_estimate[h] + reward[h] + ucb_bonus[h]
        Qt_estimate[h]= np.minimum(np.maximum(q, 0), H)
    
    episode_regrets = []
    episode_cumulative_rewards = []
    
    for _ in range(n_unrolling):

        episode_states = []
        episode_actions = []
        episode_rewards = []

        initial_state_index= 20
        state = state_space[initial_state_index]  # Initial state


        for h in range(H):
       
            state_index = np.argmin(np.abs(state_space - state))  # Find index of the current state
            q_values=Qt_estimate[h][state_index]  # Get optimistic Q-values for current state
            action_index = np.argmax(q_values)  # Find index of action with highest Q-value (greedy policy)
            action = action_space[action_index]  # Choose action with highest Q-value
            # Proceed with the chosen action
            next_state = transition_dynamics(state, action,P,state_space,action_space)
            # reward = r(state, action)
            reward=r[state_index,action_index]

            # Store observations
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state  

        episode_cum_rewards = np.sum(episode_rewards)  
        # Compute regret for the episode
        episode_regret = optimal_V[initial_state_index]- episode_cum_rewards # regret= V* - V(pi)
        episode_regrets.append(episode_regret)
        episode_cumulative_rewards.append(episode_cum_rewards)

    avg_episode_regret = np.mean(episode_regrets)
    avg_episode_cumulative_rewards = np.mean(episode_cumulative_rewards)
    # Calculate standard deviation of episode regrets and cumulative rewards
    std_episode_regret = np.std(episode_regrets)
    std_episode_cumulative_rewards = np.std(episode_cumulative_rewards)
    
    return avg_episode_regret,std_episode_regret,avg_episode_cumulative_rewards,std_episode_cumulative_rewards
