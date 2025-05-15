import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.stats import truncnorm
import wandb
import datetime
from framework import transition_dynamics, GroupInvariantKernel

def flip(x):
    return - x
group_SA = [
    lambda x: np.array([x[0], x[1]]),        # Identity for both state and action
    lambda x: np.array([flip(x[0]), flip(x[1])])  # Flip for both state and action
]

# Function for GP regression to estimate Q-values using RBF kernel
def GP_regression_invariant(X, y, state_action_space,P_kernel='RBF',l=1,alpha=1e-10):
    
    if P_kernel == "Matern_smoothness_1.5":
        kernel = GroupInvariantKernel(base_kernel="Matern", length_scale=l, smoothness=1.5, group=group_SA)
    elif P_kernel == "Matern_smoothness_2.5":
        kernel = GroupInvariantKernel(base_kernel="Matern", length_scale=l, smoothness=2.5, group=group_SA)

    elif P_kernel == "RBF":
        kernel = GroupInvariantKernel(base_kernel="RBF", length_scale=l, group=group_SA)
      
    # Define the RBF kernel

    wandb.run.summary["kernel type"] = kernel
    wandb.run.summary["alpha_gp"] = alpha
    wandb.run.summary["length_scale"] = l
    # Create GPR model
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None,alpha=alpha) 
    # Fit the model
    gpr.fit(X, y)
    # Predict mean and standard deviation
    y_pred_mean, y_pred_std = gpr.predict(state_action_space, return_std=True) 
    
    return y_pred_mean, y_pred_std


# Function for GP regression to estimate Q-values using RBF kernel
def GP_regression_with_RBF(X, y, state_action_space,l=1,alpha=1e-10):
    # Define the RBF kernel
    kernel = RBF(length_scale=l,length_scale_bounds="fixed")
    
    #tau=0.01
    wandb.run.summary["kernel type"] = kernel
    wandb.run.summary["alpha_gp"] = alpha
    wandb.run.summary["length_scale"] = l
    # Create GPR model
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None,alpha=alpha) # disabling kernel parameters optimization
    # Fit the model
    gpr.fit(X, y)
    # Predict mean and standard deviation
    y_pred_mean, y_pred_std = gpr.predict(state_action_space, return_std=True)
    
    return y_pred_mean, y_pred_std



# Main function for p-KRVI policy
def pi_krvi_policy(M, T, state_space, action_space,state_action_space, optimal_V,beta):
    S, A, H, P, r = M
    
    # Initialize wandb
    wandb.run.summary["episode length"] = H
    wandb.run.summary["episode number"] = T
    wandb.run.summary["UCB coef"] = beta

# These arrays store the observations over all episodes
    all_states=[]
    all_actions=[]
    all_rewards=[]


    Qt_mean = np.zeros((H, len(state_action_space)))
    Qt_std = np.zeros((H, len(state_action_space)))
    Qt_estimate = np.zeros((H+1, len(state_space), len(action_space)))
    for episode in range(T):
        print('episode',episode)
    
        # Initialize arrays to store observations for the current episode
        episode_states = []
        episode_actions = []
        episode_rewards = []

        initial_state_index = np.random.randint(len(state_space))  # Sample state index

        state = state_space[initial_state_index]  # Initial state
  
        # Collecting (s,a) pairs and (r+V(s')) from previous episodes
        if episode>0:
            for h in reversed(range(H)):
                X_states = np.concatenate([all_states[i][h] for i in range(episode)])
                X_actions = np.concatenate([all_actions[i][h] for i in range(episode)])
        
                # Reshape X_states and X_actions to be 2D arrays
                X_states = X_states.reshape(-1, 1)
                X_actions = X_actions.reshape(-1, 1)
        
                # Concatenate X_states and X_actions along axis 1
                X = np.concatenate((X_states, X_actions), axis=1)

                
                Qnext = []
                for i in range(episode):
                    if h < H - 1:
                        next_state_index = np.argmin(np.abs(state_space - all_states[i][h + 1]))
                        Qnext.append(np.max(Qt_estimate[h + 1][next_state_index, :]))
                    else:
                        # Handle the case where h is at the last step of the episode
                        Qnext.append(0)  # Set Qnext to 0 if we are at the last step

                y = np.array([all_rewards[i][h] + Qnext[i] for i in range(len(Qnext))])

                Qt_mean[h], Qt_std[h] = GP_regression_with_RBF(X, y, state_action_space)
                Qt_mean_reshaped = Qt_mean[h].reshape((len(state_space), len(action_space)))
                Qt_std_reshaped = Qt_std[h].reshape((len(state_space), len(action_space)))
        
                # Calculate Qt_estimate[h] from Qt_mean and Qt_std
                Qt_estimate[h] = Qt_mean_reshaped + beta * Qt_std_reshaped
          
    
        # Loop to choose actions greedily and collect observations
        for h in range(H):
            # Choose action greedily based on Q-values
            state_index = np.argmin(np.abs(state_space - state))  # Find index of the current state
            q_values=Qt_estimate[h][state_index]  # Get optimistic Q-values for current state
            action_index = np.argmax(q_values)  # Find index of action with highest Q-value
            action = action_space[action_index]  # Choose action with highest Q-value
            # Proceed with the chosen action
            next_state = transition_dynamics(state, action,P,state_space,action_space)
            reward = r[state_index, action_index]

            # Store observations
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state  #replace by next_state
        
      
        # value function of the initial state
        
        episode_cum_rewards = np.sum(episode_rewards)  # Simple approximation (negative of cumulative rewards)
         # Compute regret for the episode
        episode_regret = optimal_V[initial_state_index]- episode_cum_rewards # regret= V* - V(pi)

        metrics={"Episode_number":episode,"Episode_Regret": episode_regret, "Episode_Rewards": episode_cum_rewards}
        wandb.log(metrics)

        all_states.append(np.array(episode_states))

        all_actions.append(np.array(episode_actions))

        all_rewards.append(np.array(episode_rewards))


    plt.plot(optimal_V)
    plt.ylabel("optimal value function")
    wandb.log({"Value iteration":wandb.Image(plt)})
  


