from typing import Any, ClassVar, Optional, TypeVar, Union, Callable
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
# import wandb
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
import gpytorch
import gpytorch.settings as gpt_settings
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel, RBFKernel
import time
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import gc
os.environ["WANDB__SERVICE_WAIT"] = "300"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 
print('device',device)
from test_rotated_env import FrozenLake2DStateWrapper
from torch.profiler import profile, record_function, ProfilerActivity
from invariant_kernel import InvariantKernel, construct_90deg_block_rot_groups, apply_rotation_group
# from botorch import settings
# settings.debug(True)
torch.cuda.empty_cache()

def group_and_average(X, y):
    X = X.copy()
    y = y.copy()
    # Find unique rows and their indices
    unique_X, indices = np.unique(X, axis=0, return_inverse=True)
    
    # Create a DataFrame for y
    series_y = pd.Series(y)
    
    # Group by the indices of unique rows and average the corresponding y values
    grouped_y = series_y.groupby(indices).mean().values
    
    return unique_X, grouped_y

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

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        if self.logging:
            pass
            # wandb.init(project=logging, reinit=True, settings=wandb.Settings(start_method="thread"),mode='disabled')
            # wandb.run.summary["noise_reg"] = self.noise_reg
            # wandb.run.summary["length_scale"] = self.len_scale
            # wandb.run.summary["UCB coef"] = self.beta
            # wandb.run.summary["optim_botorch"] = self.optim_botorch
            # wandb.run.summary["seed"] = self.seed
            # wandb.run.summary["kernel"]=self.kernel
    
    

    def train(self, T: int):
        
        action_space = np.arange(self.env.action_space.n)      # Assuming discrete action space
        geometric_actions = np.array([self.action_transformation(action) for action in action_space])
        
    
        # Log hyperparameters
        if self.logging:
            pass
            # wandb.run.summary["episode length"] = self.horizon
            # wandb.run.summary["iterations"] = T
            
        # Arrays to store episode data
        
        all_states = []
        all_actions = []
        all_rewards = []
        Qt= [None] * self.horizon
        cumulative_returns = []


        for episode in range(T):
            print("Max GPU mem", torch.cuda.max_memory_allocated(1))
            gc.collect()
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
                                actions_batch = geometric_actions
                                states_expanded = np.tile(next_state, (len(action_space), 1))
                                max_q_value = 0
                                if Qt[h + 1]:
                                    # torch.cuda.memory._record_memory_history(max_entries=100000)
                                    max_q_value = np.max(
                                        self.predict_with_gp(Qt[h + 1], states_expanded, actions_batch)[0]
                                    )
                                    # try:
                                    #     torch.cuda.memory._dump_snapshot("snapshot.pickle")
                                    # except Exception as e:
                                    #     print(f"Failed to capture memory snapshot in train: {e}")
                                    # torch.cuda.memory._record_memory_history(enabled=None)

                                Qnext = max_q_value
                            else:
                                Qnext = 0

                            y_values.append(all_rewards[i][h] + Qnext)

                    if X_states:
                        X = np.column_stack((X_states, X_actions))
                        y = np.array(y_values)
                        if episode > 300:
                            X, y = group_and_average(X, y)
                        if Qt[h]:
                            Qt[h] = self.GP_regression_torch(X, y, Qt[h])
                        else:
                            Qt[h] = self.GP_regression_torch(X, y)
        
          # Execute episode
              # Initialize arrays for the current episode
            episode_states = []
            episode_actions = []
            episode_rewards = []

            initial_state, info = self.env.reset()
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
                    # try:
                    #     torch.cuda.memory._dump_snapshot("snapshot.pickle")
                    # except Exception as e:
                    #     print(f"Failed to capture memory snapshot in rollout: {e}")
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
            
            # for s in episode_states:
            #     print(s)
            # print('episode_actions',episode_actions)
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
                print(metrics)
                # wandb.log(metrics)
    
    def _GP_regression_torch(self, X, y): #I removed normalization
        """
        Gaussian Process regression using PyTorch.
        
        :param X: Input tensor of shape (n_samples, n_features)
        :param y: Target tensor of shape (n_samples,)
        :return: Trained GP model
        """
         # Ensure inputs are torch tensors and use double precision
        X = torch.tensor(X, dtype=torch.float32,device=device)
        y = torch.tensor(y, dtype=torch.float32,device=device)

        model = SingleTaskGP(train_X=X,train_Y= y.unsqueeze(-1).to(device)) #,outcome_transform=Standardize(m=1))  # GP expects (n_samples, 1) for targets
        # model.covar_module = ScaleKernel( self.kernel
        #    # RBFKernel() 
        # ).to(device) 
        model.covar_module = self.kernel.to(device)


        # Set and freeze the length scale
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [self.len_scale], dtype=torch.float32, device=device
        )
        
        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

        # Set and freeze the noise
        model.likelihood.noise = torch.tensor([self.noise_reg], dtype=torch.float32, device=device) 
        if self.optim_botorch == 0:
            model.likelihood.raw_noise.requires_grad = False

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model).to(device)
        print([param for param in model.named_parameters()])
      
        fit_gpytorch_mll(mll)
            # **Memory Cleanup**
        del X, y, mll  # Safe to delete
        torch.cuda.empty_cache()  # Free GPU memory  

        return model

    def GP_regression_torch(self, X, y, model = None): #I removed normalization
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
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x=X, train_y=y, likelihood=likelihood, kernel=self.kernel).to(device)
        else: 
            model = model.set_train_data(X, y, strict = False)
        # model = ExactGP(train_X=X,train_Y= y.unsqueeze(-1).to(device)) #,outcome_transform=Standardize(m=1))  # GP expects (n_samples, 1) for targets
        # model.covar_module = ScaleKernel( self.kernel
        #    # RBFKernel() 
        # ).to(device) 
        # model.covar_module = self.kernel.to(device)


        # Set and freeze the length scale
        # model.covar_module.base_kernel.lengthscale = torch.tensor(
        #     [self.len_scale], dtype=torch.float32, device=device
        # )
        
        # model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

        # # Set and freeze the noise
        # model.likelihood.noise = torch.tensor([self.noise_reg], dtype=torch.float32, device=device) 
        # if self.optim_botorch == 0:
        #     model.likelihood.raw_noise.requires_grad = False

        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model).to(device)
        # print([param for param in model.named_parameters()])
      
        # fit_gpytorch_mll(mll)
            # **Memory Cleanup**
        del X, y # Safe to delete
        torch.cuda.empty_cache()  # Free GPU memory  

       
        return model
    
    def predict_with_gp(self, model, states_batch, actions_batch):

        X_combined = np.hstack((states_batch, actions_batch))  # Shape: (batch_size, 2)
        X_combined = torch.tensor(X_combined, dtype=torch.float32, device=device)
        # Make predictions
        model.eval()
        with torch.no_grad():
            posterior = model(X_combined)
            mean = posterior.mean.squeeze(-1).cpu().numpy()  # Shape: (batch_size,)
            std_dev = posterior.variance.sqrt().squeeze(-1).cpu().numpy()  # Shape: (batch_size,)
        # Compute mean + beta * std_dev for each batch element
        acquisition_values = mean + self.beta * std_dev
        return acquisition_values, mean, std_dev


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# def format_bytes(size):
#     # 2**10 = 1024
#     power = 2**10
#     n = 0
#     power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
#     while size > power:
#         size /= power
#         n += 1
#     return size, power_labels[n]+'bytes'


# Example usage
if __name__ == "__main__":


    # torch.cuda.memory._record_memory_history(max_entries=100000)

    parser = argparse.ArgumentParser(description="Run KRVI Algorithm")
    # Adding arguments for user input
    parser.add_argument("--beta", type=float, default=0.1, help="UCB coefficient")
    parser.add_argument("--horizon", type=int, default=30, help="Horizon length")
    parser.add_argument("--len_scale", type=float, default=0.1, help="Length scale for GP kernel")
    parser.add_argument("--noise_reg", type=float, default=0.1, help="Noise regularization for GP")
    parser.add_argument("--env", type=str, default="FrozenLake-v1", help="Environment name")
    parser.add_argument("--logging", type=str, default="IQL_project", help="wandb project name")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info)")
    parser.add_argument("--iterations", type=int, default=3000, help="Number of training iterations (T)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--optim_botorch", type= int, default = 1, help ='turn on hyperparm optimization by botorch')



    args = parser.parse_args()

    env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    env = FrozenLake2DStateWrapper(env, rescale=True)
    optimal_V= None
    # k_G = InvariantKernel(
    #     base_kernel=RBFKernel(),
    #     transformations=apply_rotation_group,
    #     is_isotropic=True,
    #     is_group=True,
    # )
    k_G = InvariantKernel(
    base_kernel=RBFKernel(),
    transformations=apply_rotation_group,
    is_isotropic=True,
    is_group=True,
    )
    # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    krvi = KRVI(
        kernel = k_G,
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
    # with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
    krvi.train(T= args.iterations)

    # prof.export_chrome_trace("trace3.json")
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    # try:
    #     torch.cuda.memory._dump_snapshot("snapshot.pickle")
    # except Exception as e:
    #     print(f"Failed to capture memory snapshot in train: {e}")

    # torch.cuda.memory._record_memory_history(enabled=None)





