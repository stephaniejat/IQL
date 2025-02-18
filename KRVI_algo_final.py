from typing import Any, ClassVar, Optional, TypeVar, Union, Callable
import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
import gpytorch
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel, RBFKernel
import time
import warnings
warnings.filterwarnings("ignore")
import os
import csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print('device',device)



    
def preprocess_state(state):

    if isinstance(state, dict):
        # Extract the observation from the dictionary
        state = state['observation']
    elif isinstance(state, tuple):
        # Convert the tuple to a numpy array
        state = np.array(state)
    elif isinstance(state, (int, float)):
        # Convert scalar to a 1D numpy array
        state = np.array([state])
    elif isinstance(state, np.ndarray):
        # Ensure the state is a numpy array
        state = state
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")
    if len(state.shape) > 1:
        state = state.flatten()
    
    return state


class KRVI:
    def __init__(
        self,
        kernel:  Callable,
        env: Union[gym.Env, str],
        beta: float,
        horizon: int,
        action_transformation: Callable,
        len_scale: float = 0.1,
        noise_reg: float = 0.5,
        optim_botorch: int = 0,
        optimal_V: Optional[np.ndarray] = None,
        logging: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self.kernel = kernel
        self.env = gym.make(env) if isinstance(env, str) else env
        self.beta = beta
        self.len_scale = len_scale
        self.noise_reg = noise_reg
        self.horizon = horizon
        self.logging = logging
        self.verbose = verbose
        self.seed = seed
        self.optim_botorch = optim_botorch
        self.optimal_V = optimal_V
        self.action_transformation = action_transformation
        # File paths
        self.csv_file = 'krvi_metrics.csv'
        self.config_file = 'config.txt'

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        if self.logging:
            with open(self.config_file, mode='w') as f:
                f.write(f"beta={self.beta}\n")
                f.write(f"len_scale={self.len_scale}\n")
                f.write(f"noise_reg={self.noise_reg}\n")
                f.write(f"horizon={self.horizon}\n")
                f.write(f"seed={self.seed}\n")
                f.write(f"optim_botorch={self.optim_botorch}\n")

            # Write headers to the metrics CSV file if it doesn't exist
            if not os.path.exists(self.csv_file):
                with open(self.csv_file, mode='w', newline='') as f:
                    print('hey')
                    writer = csv.writer(f)
                    writer.writerow(['Episode', 'Reward', 'Cumulative Returns'])  # Column headers

           
            
    

    def train(self, T: int):
        
        action_space = np.arange(self.env.action_space.n)      # Assuming discrete action space
       
            
        # Arrays to store episode data
        all_states = []
        all_actions = []
        all_rewards = []
        Qt= [None] * self.horizon
        cumulative_returns = []


        for episode in range(T):
            if self.verbose > 0:
                print(f'Episode {episode}')
                # Update Q-values from previous episodes
            if episode > 0:
                for h in reversed(range(len(all_states[-1]))):
                    X_states = []
                    X_actions = []
                    y_values = []

                    for i in range(episode):
                        if h < len(all_states[i]):
                            X_states.append(all_states[i][h])
                            X_actions.append(all_actions[i][h])

                            if h < len(all_states[i]) - 1:
                                next_state = all_states[i][h + 1]
                                actions_batch = np.array([self.action_transformation(action) for action in action_space])
                                states_expanded = np.tile(next_state, (len(action_space), 1))
                                max_q_value = 0
                                if Qt[h + 1]:
                                    max_q_value = np.max(
                                        self.predict_with_gp(Qt[h + 1], states_expanded, actions_batch)[0]
                                    )
                                Qnext = max_q_value
                            else:
                                Qnext = 0

                            y_values.append(all_rewards[i][h] + Qnext)

                    if X_states:
                        X = np.column_stack((X_states, X_actions))
                        y = np.array(y_values)
                        Qt[h] = self.GP_regression_torch(X, y)

        
          # Execute episode
              # Initialize arrays for the current episode
            episode_states = []
            episode_actions = []
            episode_rewards = []

            initial_state, info = self.env.reset()
            state= preprocess_state(initial_state) 

          
           
            for h in range(self.horizon):
            
                if Qt[h]:  # Ensure a model is available for the current step
               
                    # Prepare inputs for batched prediction
                    states_batch = np.tile(state, (len(action_space), 1))
                    actions_batch = np.array([self.action_transformation(action) for action in action_space])
                    # Predict Q-values for all actions in a single batch
                    q_values = self.predict_with_gp(Qt[h], states_batch, actions_batch)[0]

                else:
                    # Default Q-values if no model is available
                    q_values = np.zeros(len(action_space))

                # Select action with the highest Q-value
            
                action = action_space[np.argmax(q_values)]
                next_state, reward, done, truncated , info = self.env.step(action)
                next_state = preprocess_state(next_state)  # Convert to numpy array
               
            
                episode_states.append(state)
                action= self.action_transformation(action)
                episode_actions.append(action)
                episode_rewards.append(reward)
                  # If done or truncated, break the loop early
                if done or truncated:
                    break
          
                state = next_state
            

            all_states.append(np.array(episode_states))
            all_actions.append(np.array(episode_actions))
            all_rewards.append(np.array(episode_rewards))


            episode_cum_rewards = np.sum(episode_rewards)
            cumulative_returns.append(episode_cum_rewards) 


            if self.logging:
                 # Log to CSV file
                with open(self.csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, episode_cum_rewards, sum(cumulative_returns)])
    
    def GP_regression_torch(self, X, y): #I removed normalization
        """
        Gaussian Process regression using PyTorch.
        
        :param X: Input tensor of shape (n_samples, n_features)
        :param y: Target tensor of shape (n_samples,)
        :return: Trained GP model
        """
         # Ensure inputs are torch tensors and use double precision
        X = torch.tensor(X, dtype=torch.float64,device=device)
        y = torch.tensor(y, dtype=torch.float64,device=device)

        model = SingleTaskGP(train_X=X,train_Y= y.unsqueeze(-1).to(device)) #,outcome_transform=Standardize(m=1))  # GP expects (n_samples, 1) for targets
        model.covar_module = ScaleKernel( self.kernel
           # RBFKernel() 
        ).to(device) 


# Set and freeze the length scale
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [self.len_scale], dtype=torch.float64, device=device
        )
        
        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

        # Set and freeze the noise
        model.likelihood.noise = torch.tensor([self.noise_reg], dtype=torch.float64, device=device) 
        if self.optim_botorch == 0:
            model.likelihood.raw_noise.requires_grad = False

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model).to(device)
       
      
        fit_gpytorch_mll(mll)

       
        return model
    
    def predict_with_gp(self, model, states_batch, actions_batch):

        X_combined = np.hstack((states_batch, actions_batch))  # Shape: (batch_size, 2)
        X_combined = torch.tensor(X_combined, dtype=torch.float64, device=device)
        # Make predictions
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_combined)
            mean = posterior.mean.squeeze(-1).cpu().numpy()  # Shape: (batch_size,)
            std_dev = posterior.variance.sqrt().squeeze(-1).cpu().numpy()  # Shape: (batch_size,)
        # Compute mean + beta * std_dev for each batch element
        acquisition_values = mean + self.beta * std_dev
        return acquisition_values, mean, std_dev


 
