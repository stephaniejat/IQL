import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, Kernel
from scipy.stats import truncnorm
import os
import wandb
import datetime
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import datetime
from itertools import product

# #kernel implementation from Scikit learn
class GroupInvariantKernel(Kernel):
    def __init__(self, base_kernel="Matern", length_scale=0.1, smoothness=1.5, group=None):
        self.base_kernel = base_kernel
        self.length_scale = length_scale
        self.smoothness = smoothness
        self.group = group if group is not None else [lambda x: x]  # Default to identity transformation if no group given

        # Define the base kernel as Matern or RBF
        if self.base_kernel == "Matern":
            self.kernel = Matern(length_scale=length_scale, nu=smoothness)
        elif self.base_kernel == "RBF":
            self.kernel = RBF(length_scale=length_scale)

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.group_invariant_kernel(X, Y,eval_gradient=eval_gradient)

    def group_invariant_kernel(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
       # print('X',X)
        Y = np.atleast_2d(Y)
        # Ensure the dimensions are as expected
       
       # print('Y',Y)
        # Calculate the kernel averaged over the group transformations
        K = np.zeros((X.shape[0], Y.shape[0]))

        # # Loop over all transformations in the group
        for g in self.group:
            #print('g',g)
            X_transformed = np.array([g(x) for x in X])
            #print('X_transformed',X_transformed.shape)
            # print('X_transformed values',X_transformed)
            # print('Y',Y.shape)
             # Check if the kernel is producing a valid square matrix for each transformation
            transformed_kernel = self.kernel(X_transformed, Y)
            #print('Transformed kernel shape:', transformed_kernel.shape)
            
            K += transformed_kernel / len(self.group)
    


        return K

    def diag(self, X):
        return np.diag(self.__call__(X))
  

    def is_stationary(self):
        return True
        #return self.kernel.is_stationary()
#
def shift_05(x):
    return (x + 0.5) % 1

def shift_03(x):
    return (x + 0.3) % 1
def square(x):
    return x**2

def flip(x):
    return - x


group_SA = [
    lambda x: np.array([x[0], x[1]]),        # Identity for both state and action
    lambda x: np.array([flip(x[0]), flip(x[1])])  # Flip for both state and action
]




group_SASp = [
    lambda x: np.array([x[0],x[1],x[2]]),        # Identity for both state and action
    lambda x: np.array([flip(x[0]), flip(x[1]), flip(x[2])])  # Flip for both state and action
]



# Define the reward_RKHS function with the group-invariant kernel
def reward_RKHS(P_kernel, state_space, action_space, subdir=None, alpha=0.5):
    grid_size = 5  # Grid size for fitting GP regression

    # Generate all possible input points in the grid
    values = np.linspace(-1, 1, grid_size)
    X = np.array(list(product(values, repeat=2)))  # 2D grid points for state-action pairs

    # Define the group-invariant kernel based on the P_kernel parameter and group G (state-action transformations)
    if P_kernel == "Matern_smoothness_1.5":
        kernel = GroupInvariantKernel(base_kernel="Matern", length_scale=0.1, smoothness=1.5, group=group_SA)
    elif P_kernel == "Matern_smoothness_2.5":
        kernel = GroupInvariantKernel(base_kernel="Matern", length_scale=0.1, smoothness=2.5, group=group_SA)

    elif P_kernel == "RBF":
        kernel = GroupInvariantKernel(base_kernel="RBF", length_scale=0.1, group=group_SA)

    # Sample y values from the Gaussian process with the group-invariant kernel
    gp = GaussianProcessRegressor(kernel=kernel)

    y = gp.sample_y(X, 1)

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=alpha)
    K = gpr.kernel(X,X)
    eigvals = np.linalg.eigvalsh(K)

    gpr.fit(X, y)

    values = np.linspace(-1, 1, len(state_space))
    all_possible_inputs = np.array(list(product(values, repeat=2)))  # State-action pairs in 2D

    # Predict for all possible input points
    all_predictions, _ = gpr.predict(all_possible_inputs, return_std=True)
    y_pred, _ = gpr.predict(X, return_std=True)
    mse = mean_squared_error(y, y_pred)
    # Scale and normalize the predictions
    min_prediction = np.min(all_predictions)
    max_prediction = np.max(all_predictions)
    scaled_predictions = (all_predictions - min_prediction) / (max_prediction - min_prediction)
    r = scaled_predictions.reshape((len(state_space), len(action_space)))

    return r

def transition_P_RKHS(state_space, action_space, P_kernel, alpha=0.5):
    grid_size = 5  # Grid size for fitting GP regression
    values = np.linspace(-1, 1, grid_size)
    X = np.array(list(product(values, repeat=3)))  # 3D points for state-action-next_state
   
    # # Define the group-invariant kernel based on the P_kernel parameter and transformations for 3D inputs
    if P_kernel == "Matern_smoothness_1.5":
        kernel = GroupInvariantKernel(base_kernel="Matern", length_scale=0.1, smoothness=1.5, group=group_SASp)
    elif P_kernel == "Matern_smoothness_2.5":
        kernel = GroupInvariantKernel(base_kernel="Matern", length_scale=0.1, smoothness=2.5, group=group_SASp)
    elif P_kernel == "RBF":
        kernel = GroupInvariantKernel(base_kernel="RBF", length_scale=0.1, group=group_SASp)

    # Sample y values from the Gaussian process
    gp = GaussianProcessRegressor(kernel=kernel)
    y = gp.sample_y(X, 1)

    # Fit the Gaussian Process Regressor
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=alpha)
    gpr.fit(X, y)
    print('after fitting')

    # Generate all possible input points for prediction (state-action-next_state)
    values = np.linspace(-1, 1, len(state_space))
    all_possible_inputs = np.array(list(product(values, repeat=3)))  # 3D points for state-action-next_state

    # Predict for all possible input points
    all_predictions, _ = gpr.predict(all_possible_inputs, return_std=True)

    # Scale up and normalize the predictions
    min_prediction = np.min(all_predictions)
    max_prediction = np.max(all_predictions)
    scaled_predictions = (all_predictions - min_prediction) / (max_prediction - min_prediction)

    num_state_action_pairs = len(all_possible_inputs) // len(state_space)
    for i in range(num_state_action_pairs):
        state_start_index = i * len(state_space)
        state_end_index = (i + 1) * len(state_space)
        state_predictions = scaled_predictions[state_start_index:state_end_index]
        scaled_predictions[state_start_index:state_end_index] = state_predictions / np.sum(state_predictions)

    # Reshape the predictions to form the transition probability tensor P
    P = scaled_predictions.reshape((len(state_space), len(action_space), len(state_space)))
    print('P,', P)
    return P


# Define transition dynamics function which takes current state and action values as input and outputs the value of the next state
def transition_dynamics(current_state, action, transition_P,state_space,action_space): 

    # Get the index of the current state
    current_state_idx = np.argmin(np.abs(state_space - current_state))
    # Get the index of the action
    action_idx = np.argmin(np.abs(action_space - action))

    # Retrieve transition probabilities from the precomputed dynamics
    transition_probs = transition_P[current_state_idx, action_idx, :]

    # Sample next state based on transition probabilities
    next_state_idx = np.random.choice(np.arange(len(state_space)), p=transition_probs)

    next_state = state_space[next_state_idx]

    return next_state
   

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
                    reward = r[s_idx,a_idx]
                    expected_return += P[s_idx, a_idx, next_s_idx] * (reward + V[next_s_idx])
                Q_values.append(expected_return)
            V_new[s_idx] = max(Q_values)  # Update value function for state s
        V = V_new  # Update value function after each iteration

    return V # V is indexed by state index


def plot_reward_gp_3d(r, state_space, action_space, save_dir=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    action_mesh, state_mesh = np.meshgrid(state_space, action_space)
    surf = ax.plot_surface(state_mesh, action_mesh, r, cmap='viridis',vmin=0,vmax=1)
    fig.colorbar(surf, shrink=0.5, aspect=5)


    
    ax.set_xlabel('s',fontsize=20)
    ax.set_ylabel('a',fontsize=20)
    ax.set_zlabel('r(s,a)',fontsize=20)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(save_dir, f"reward_gp_3d_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()
    else:
        wandb.log({"Reward Surface 3D": wandb.Image(plt)})
    
   

def plot_transition_probabilities(P, state_space,action_space, save_dir=None):


    
    state_indices = [0, 5, 9]  # Indices of states in state_space
    action_indices = [0, 5, 9]  # Indices of actions in action_space

# Plot transition probabilities for each (s, a) pair
    for state_idx in state_indices:
        for action_idx in action_indices:
    # Get the transition probabilities for the specified state-action pair
            transition_probs = P[state_idx, action_idx, :]

            state_labels = [str(round(float(state[0]), 4)) for state in state_space]
            action_labels = [str(round(float(action[0]), 4)) for action in action_space]
            
            # Plot
            plt.figure(figsize=(8, 6))
            plt.bar(state_labels, transition_probs)
            plt.xlabel("s'",fontsize=20)
            # plt.rcParams['text.usetex'] = True
            # plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
            
            plt.ylabel(r"$P(s' \mid s={}, a={})$".format(state_labels[state_idx], action_labels[action_idx]), fontsize=20)
            
            plt.xticks([state_labels[0], state_labels[-1]])
            
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"P_s{state_idx}_a{action_idx}.png")
                plt.savefig(save_path)  # Save the figure with a unique name
                plt.close()
            else: 
                # Convert the Matplotlib figure to a wandb Image and log it
                wandb.log({f"Transition Probabilities for State-Action Pair (s={state_space[state_idx]}, a={action_space[action_idx]})": wandb.Image(plt)})

