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
        #print('length_scale',l)
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
    y_pred_mean, y_pred_std = gpr.predict(state_action_space, return_std=True)  # We should not evaluate at discrete state-action space, return gpr
    
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
    #gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None) #correct
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None,alpha=alpha) # disabling kernel parameters optimization
    # Fit the model
    gpr.fit(X, y)
    # Predict mean and standard deviation
    y_pred_mean, y_pred_std = gpr.predict(state_action_space, return_std=True)
    
    return y_pred_mean, y_pred_std

# # Function for GP regression to estimate Q-values using Matérn kernel
# def GP_regression_with_Matern(X, y, state_action_space, alpha=1.0, nu=1.5):
#     # Define the Matérn kernel
#     kernel = Matern(length_scale=1.0, length_scale_bounds="fixed", nu=nu)
#     wandb.run.summary["kernel type"] = kernel
#     # Create GPR model
#     gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=None)
#     # Fit the model
#     gpr.fit(X, y)
#     # Predict mean and standard deviation
#     y_pred_mean, y_pred_std = gpr.predict(state_action_space, return_std=True)

#     return y_pred_mean, y_pred_std


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
        #initial_state_index=2
        ###print('initial state index',initial_state_index)
        state = state_space[initial_state_index]  # Initial state
  
        # Collecting (s,a) pairs and (r+V(s')) from previous episodes
        if episode>0:
            ## complete the code
            for h in reversed(range(H)):
                ###print('h',h)
                X_states = np.concatenate([all_states[i][h] for i in range(episode)])
                X_actions = np.concatenate([all_actions[i][h] for i in range(episode)])
        
                # Reshape X_states and X_actions to be 2D arrays
                X_states = X_states.reshape(-1, 1)
                ###print('concatinated states',X_states)
                X_actions = X_actions.reshape(-1, 1)
                ###print('concatinated actions',X_actions)
        
                # Concatenate X_states and X_actions along axis 1
                X = np.concatenate((X_states, X_actions), axis=1)
                #print('X',X)
                ###print('concatinated rewards',np.concatenate([np.array(all_rewards[i][h]) for i in range(episode)]))
                
                Qnext = []
                for i in range(episode):
                    if h < H - 1:
                        next_state_index = np.argmin(np.abs(state_space - all_states[i][h + 1]))
                        ###print('next_state_index',next_state_index)
                        Qnext.append(np.max(Qt_estimate[h + 1][next_state_index, :]))
                    else:
                        # Handle the case where h is at the last step of the episode
                        Qnext.append(0)  # Set Qnext to 0 if we are at the last step
                ###print('Qnext',Qnext)
                # Concatenate rewards and Qnext along axis 0 and add them element-wise
                #print('all_rewards',all_rewards)
                #print('Qnext',Qnext)
                y = np.array([all_rewards[i][h] + Qnext[i] for i in range(len(Qnext))])
                #print('y',y)
                #y = np.concatenate([np.array(all_rewards[i][h]) + Qnext[i] for i in range(len(Qnext))])
                ###print('y',y)
                Qt_mean[h], Qt_std[h] = GP_regression_with_RBF(X, y, state_action_space)
                ###print('Qt_mean',Qt_mean[h])
                ###print('Qt_std',Qt_std[h])
                Qt_mean_reshaped = Qt_mean[h].reshape((len(state_space), len(action_space)))
                Qt_std_reshaped = Qt_std[h].reshape((len(state_space), len(action_space)))
        
                # Calculate Qt_estimate[h] from Qt_mean and Qt_std
                Qt_estimate[h] = Qt_mean_reshaped + beta * Qt_std_reshaped
                ###print('Qt_estimate[h]',Qt_estimate[h])
          
    
        # Loop to choose actions greedily and collect observations
        for h in range(H):
            # Choose action greedily based on Q-values
            state_index = np.argmin(np.abs(state_space - state))  # Find index of the current state
            ###print('state_index',state_index)
            q_values=Qt_estimate[h][state_index]  # Get optimistic Q-values for current state
            ###print('q_values',q_values)
            action_index = np.argmax(q_values)  # Find index of action with highest Q-value
            action = action_space[action_index]  # Choose action with highest Q-value
            ###print('action',action)
            # Proceed with the chosen action
            next_state = transition_dynamics(state, action,P,state_space,action_space)
            reward = r[state_index, action_index]

            # Store observations
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            #state_index = np.argmin(np.abs(state_space - next_state))  # Find closest next state index
            state = next_state  #replace by next_state
        
      
        # value function of the initial state
        
        #print('Optimal V value iteration',optimal_value_function[initial_state_index])
        episode_cum_rewards = np.sum(episode_rewards)  # Simple approximation (negative of cumulative rewards)
         # Compute regret for the episode
        episode_regret = optimal_V[initial_state_index]- episode_cum_rewards # regret= V* - V(pi)
        #print('episode_regret',episode_regret)
        #print('episode cum rewards',episode_cum_rewards)
        metrics={"Episode_number":episode,"Episode_Regret": episode_regret, "Episode_Rewards": episode_cum_rewards}
        wandb.log(metrics)
        #cum_rewards.append( episode_cum_rewards)
        ##print('episode_states',episode_states)
        all_states.append(np.array(episode_states))
        #print('all_states',all_states)
        ##print('episode_actions',episode_actions)
        all_actions.append(np.array(episode_actions))
        #print('all_actions',all_actions)
        ##print('episode_rewards',episode_rewards)
        all_rewards.append(np.array(episode_rewards))
        ##print('all_rewards',all_rewards)
        # At the end of the function, inside the loop where episodes are being iterated
        
        # if episode == T - 1:  # Check if it's the last episode
        #     # Convert Qt_estimate to a table format
        #     table_data = []
        #     for h in range(H):
        #         for i, state in enumerate(state_space):
        #             for j, action in enumerate(action_space):
        #                 table_data.append([h, state, action, Qt_estimate[h][i][j]])
        #     # Log the table
            #wandb.log({"Qt_estimate_table": wandb.Table(data=table_data, columns=["Step", "State", "Action", "Q_estimate"])})

    plt.plot(optimal_V)
    plt.ylabel("optimal value function")
    wandb.log({"Value iteration":wandb.Image(plt)})
  


