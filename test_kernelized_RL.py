import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.stats import truncnorm
import wandb
import datetime


# Define a function to generate transition dynamics
def transition_P(state_space,action_space):
    std_dev = 0.1  # Standard deviation of the Gaussian distribution
    clip_min = 0  # Minimum value to clip the distribution
    clip_max = 1  # Maximum value to clip the distribution
    # Define number of buckets to discretize the probability distribution
    num_buckets = len(state_space)

    # Transition probability matrix P(s, a, s')
    P = np.zeros((state_space.shape[0], action_space.shape[0], state_space.shape[0]))

    # Compute transition probabilities
    for s_idx, s in enumerate(state_space):
        for a_idx, a in enumerate(action_space):
            # Calculate the mean of the next state
            next_state_mean = 2 * s - a
            # Clip next_state_mean to ensure it's within the state_space
            next_state_mean = np.clip(next_state_mean, clip_min, clip_max)
            # Quantize next_state_mean to the nearest value in state_space
            next_state_mean_idx = np.argmin(np.abs(state_space - next_state_mean))
            next_state_mean = state_space[next_state_mean_idx]
            # Calculate the bounds for truncation of the normal distribution
            a_clip = (clip_min - next_state_mean) / std_dev
            b_clip = (clip_max - next_state_mean) / std_dev
            
            # Generate samples from the truncated normal distribution
            samples = truncnorm.rvs(a=a_clip, b=b_clip, loc=next_state_mean, scale=std_dev, size=1000)
            
            # Count occurrences of samples within each bucket
            bucket_counts = np.zeros(num_buckets)
            for sample in samples:
                bucket_idx = int(np.floor(sample * num_buckets))
                bucket_counts[bucket_idx] += 1

           
            # Normalize bucket counts to obtain probabilities
            bucket_probs = bucket_counts / np.sum(bucket_counts)
            
            # Assign bucket probabilities to corresponding discrete states
            P[s_idx, a_idx, :] = bucket_probs
    
    return P # P is indexed by the state and action indices not values
# Define transition dynamics function which takes current state and action values as input and outputs the value of the next state
def transition_dynamics(current_state, action, transition_P,state_space,action_space): 

    # Get the index of the current state
    current_state_idx = np.argmin(np.abs(state_space - current_state))
    # Get the index of the action
    action_idx = np.argmin(np.abs(action_space - action))

    # Retrieve transition probabilities from the precomputed dynamics
    transition_probs = transition_P[current_state_idx, action_idx, :]
    # print(transition_probs)
    # print(np.arange(len(state_space)))
    # Sample next state based on transition probabilities
    next_state_idx = np.random.choice(np.arange(len(state_space)), p=transition_probs)
    # print(next_state_idx)

    next_state = state_space[next_state_idx]

    return next_state
   
    # Define reward function
def reward_function(state, action): #reward function takes state and action values
    reward = np.sin(2 * np.pi * state) * np.cos(2 * np.pi * action)  # Example sinusoidal reward
    # Normalize the reward to be between 0 and 1
    reward = (reward + 1) / 2

    return reward


# Value iteration algorithm
def value_iteration_episodic(state_space, action_space, r, P, H=10):
    V = np.zeros_like(state_space)  # Initialize value function
    for h in range(H):  # Iterate over time steps
        V_new = np.zeros_like(V)  # Initialize new value function
        for s_idx, s in enumerate(state_space):
            Q_values = []
            for a_idx, a in enumerate(action_space):
                expected_return = 0
                for next_s_idx, next_s in enumerate(state_space):
                    reward = r(s,a)
                    #reward=reward_function_RKHS(s,a,r)
                    expected_return += P[s_idx, a_idx, next_s_idx] * (reward + V[next_s_idx])
                Q_values.append(expected_return)
            V_new[s_idx] = max(Q_values)  # Update value function for state s
        V = V_new  # Update value function after each iteration

    return V # V is indexed by state index




# Function for GP regression to estimate Q-values using RBF kernel
def GP_regression_with_RBF(X, y, state_action_space):
    # Define the RBF kernel
    kernel = RBF(length_scale_bounds="fixed")
    wandb.run.summary["kernel type"] = kernel
    # Create GPR model
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None) #using default alpha 1e-10 and disabling kernel parameters optimization
    # Fit the model
    gpr.fit(X, y)
    # Predict mean and standard deviation
    y_pred_mean, y_pred_std = gpr.predict(state_action_space, return_std=True)
    
    return y_pred_mean, y_pred_std

# Function for GP regression to estimate Q-values using Matérn kernel
def GP_regression_with_Matern(X, y, state_action_space, alpha=1.0, nu=1.5):
    # Define the Matérn kernel
    kernel = Matern(length_scale=1.0, length_scale_bounds="fixed", nu=nu)
    wandb.run.summary["kernel type"] = kernel
    # Create GPR model
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=None)
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
                ###print('X',X)
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
                y = np.concatenate([np.array(all_rewards[i][h]) + Qnext[i] for i in range(len(Qnext))])
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
            reward = r(state, action)

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
        ##print('all_states',all_states)
        ##print('episode_actions',episode_actions)
        all_actions.append(np.array(episode_actions))
        ##print('all_actions',all_actions)
        ##print('episode_rewards',episode_rewards)
        all_rewards.append(np.array(episode_rewards))
        ##print('all_rewards',all_rewards)
        # At the end of the function, inside the loop where episodes are being iterated
        
        if episode == T - 1:  # Check if it's the last episode
            # Convert Qt_estimate to a table format
            table_data = []
            for h in range(H):
                for i, state in enumerate(state_space):
                    for j, action in enumerate(action_space):
                        table_data.append([h, state, action, Qt_estimate[h][i][j]])
            # Log the table
            wandb.log({"Qt_estimate_table": wandb.Table(data=table_data, columns=["Step", "State", "Action", "Q_estimate"])})

    plt.plot(optimal_V)
    plt.ylabel("optimal value function")
    wandb.log({"Value iteration":wandb.Image(plt)})
  


