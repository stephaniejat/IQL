from typing import Any, ClassVar, Optional, TypeVar, Union, Callable
import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import wandb
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
import gpytorch
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel, RBFKernel
import time
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "300"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
print('device',device)
from test_rotated_reflected import FrozenLake2DStateWrapper
#from test_rotated_env import FrozenLake2DStateWrapper
from invariant_kernel import InvariantKernel, construct_90deg_block_rot_groups, apply_rotation_group
import csv

def action_transformation(action_index):
    action_map = {
        0: np.array([-1, 0]),  # Left
        1: np.array([0, -1]),  # Down
        2: np.array([1, 0]),   # Right
        3: np.array([0, 1])    # Up
    }
    return action_map.get(action_index, np.array([0, 0]))  # Default to [0, 0] if invalid index
    

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
        self.csv_file = 'krvi_metrics.csv'
        self.config_file = 'config.txt'

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        if self.logging:
           
            #wandb.init(mode='disabled')
            wandb.init(project=logging, reinit=True, settings=wandb.Settings(start_method="thread"))
            wandb.run.summary["noise_reg"] = self.noise_reg
            wandb.run.summary["length_scale"] = self.len_scale
            wandb.run.summary["UCB coef"] = self.beta
            wandb.run.summary["optim_botorch"] = self.optim_botorch
            wandb.run.summary["seed"] = self.seed
            wandb.run.summary["kernel"]= self.kernel

            # with open(self.config_file, mode='w') as f:
            #     f.write(f"beta={self.beta}\n")
            #     f.write(f"len_scale={self.len_scale}\n")
            #     f.write(f"noise_reg={self.noise_reg}\n")
            #     f.write(f"horizon={self.horizon}\n")
            #     f.write(f"seed={self.seed}\n")
            #     f.write(f"optim_botorch={self.optim_botorch}\n")
            #     f.write(f"kernel={self.kernel}\n")

            # # Write headers to the metrics CSV file if it doesn't exist
            # if not os.path.exists(self.csv_file):
            #     with open(self.csv_file, mode='w', newline='') as f:
                    
            #         writer = csv.writer(f)
            #         writer.writerow(['Episode', 'Reward', 'Cumulative Returns'])  # Column headers
    
    

    def train(self, T: int):
        
        action_space = np.arange(self.env.action_space.n)      # Assuming discrete action space
       
        
    
        # Log hyperparameters
        if self.logging:
            wandb.run.summary["episode length"] = self.horizon
            wandb.run.summary["iterations"] = T
            
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
                    print('h',h)
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
                        
                        if Qt[h]:
                            Qt[h] = self.GP_regression_torch(X[-1].reshape(1,-1), y[-1].reshape(1,-1), Qt[h])
                        else:
                            Qt[h] = self.GP_regression_torch(X, y, None)
 

        
          # Execute episode
              # Initialize arrays for the current episode
            episode_states = []
            episode_actions = []
            episode_rewards = []

            initial_state, info = self.env.reset()
            # print('initial_state',initial_state)
            #print('self.current_rotation',self.env.current_rotation)
            #print('self.desc',self.env.desc)

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
                metrics = {
                    "Episode_number": episode,
                    "Episode_Rewards": episode_cum_rewards,
                    "cumulative_returns": sum(cumulative_returns)
                }
                wandb.log(metrics)
                # with open(self.csv_file, mode='a', newline='') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([episode, episode_cum_rewards, sum(cumulative_returns)])

    
    def GP_regression_torch(self, X, y, model): #I removed normalization
            """
            Gaussian Process regression using PyTorch.
            
            :param X: Input tensor of shape (n_samples, n_features)
            :param y: Target tensor of shape (n_samples,)
            :return: Trained GP model
            """
            # Ensure inputs are torch tensors and use double precision
            X = torch.tensor(X, dtype=torch.float32,device=device)
            y = torch.tensor(y, dtype=torch.float32,device=device)
            if model is None:
                model = SingleTaskGP(train_X=X,train_Y= y.unsqueeze(-1).to(device), covar_module = self.kernel.to(device)) #,outcome_transform=Standardize(m=1))  # GP expects (n_samples, 1) for targets
                lengthscale_tensor = torch.tensor([self.len_scale], dtype=torch.float32, device=device)

                if hasattr(model.covar_module, 'base_kernel'):
                    # model.covar_module.base_kernel works only for the invariant kernel
                    base_covar = model.covar_module.base_kernel
                else:
                    base_covar = model.covar_module
                # Set and freeze the length scale
                base_covar.lengthscale = lengthscale_tensor
                base_covar.raw_lengthscale.requires_grad = False
            else:
                model = model.condition_on_observations(X=X, Y=y)

            if self.optim_botorch == 1:
                
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model).to(device)
                # for param_name, param in model.named_parameters():
                #     print(f'Parameter name: {param_name:42} value before fitting = {param.item()} requires_grad: {param.requires_grad}')


                with gpytorch.settings.cholesky_max_tries(6):
                    fit_gpytorch_mll(mll)
                # for param_name, param in model.named_parameters():
                #     print(f'Parameter name: {param_name:42} value after fitting = {param.item()} requires_grad: {param.requires_grad}')

            torch.cuda.empty_cache()  # Free GPU memory  

        
            return model
        
    def predict_with_gp(self, model, states_batch, actions_batch):

        X_combined = np.hstack((states_batch, actions_batch))  # Shape: (batch_size, 2)
        X_combined = torch.tensor(X_combined, dtype=torch.float32, device=device)
        # Make predictions
        model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            posterior = model.posterior(X_combined)
            mean = posterior.mean.squeeze(-1).detach().cpu().numpy()  # Shape: (batch_size,)
            std_dev = posterior.variance.sqrt().squeeze(-1).detach().cpu().numpy()  # Shape: (batch_size,)
        # Compute mean + beta * std_dev for each batch element
        acquisition_values = mean + self.beta * std_dev
        return acquisition_values, mean, std_dev


 
# Example usage
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run KRVI Algorithm")
    # Adding arguments for user input
    parser.add_argument("--beta", type=float, default=0.1, help="UCB coefficient")
    parser.add_argument("--horizon", type=int, default=100, help="Horizon length")
    parser.add_argument("--len_scale", type=float, default=0.1, help="Length scale for GP kernel")
    parser.add_argument("--noise_reg", type=float, default=0.1, help="Noise regularization for GP")
    parser.add_argument("--env", type=str, default="FrozenLake-v1", help="Environment name")
    parser.add_argument("--logging", type=str, default="trial", help="wandb project name") #IQL_project_invariant
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info)")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of training iterations (T)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--optim_botorch", type= int, default = 1, help ='turn on hyperparm optimization by botorch')
    parser.add_argument("--kernel", type=str, default="invariant_kernel", help="Choose the kernel between invariant kernel and RBF kernel")




    args = parser.parse_args()

    env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    env = FrozenLake2DStateWrapper(env, rescale=True)
    # print(env.desc)
    optimal_V= None
    if args.kernel=='RBF':
        k_G=RBFKernel()
    elif args.kernel == 'invariant_kernel':

        k_G = InvariantKernel(
        base_kernel=RBFKernel(),
        transformations=apply_rotation_group,
        is_isotropic=True,
        is_group=True,
        )



    krvi = KRVI(
        kernel= k_G,
        env= env,
        beta=args.beta,
        horizon=args.horizon,
        action_transformation = action_transformation,
        len_scale=args.len_scale,
        noise_reg=args.noise_reg,
        optim_botorch=args.optim_botorch,
        optimal_V = optimal_V,
        logging=args.logging,
        verbose=args.verbose,
        seed= args.seed
    )

    krvi.train(T= args.iterations)

















