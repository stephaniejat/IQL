from typing import Any, ClassVar, Optional, TypeVar, Union, Callable

import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
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
os.environ["WANDB__SERVICE_WAIT"] = "300"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print('device',device)


class KRVI:
    """
    KRVI

    Paper: "https://"

    
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param backend: the algorithm used to perform KRR when fitting Q
    :param kernel: the kernel used
    :param beta: UCB factor
    :param horizon: horizon to run
   
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    
    :param logging: the log location (if None, no logging)
    
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    
    
    """


    def __init__(
        self,
        kernel: Union[str, type],
        backend: Union[str, type],
        env: Union[gym.Env, str],
        beta: float,
        horizon: int,
        len_scale: float = 0.1,  # Added parameter for length scale
        noise_reg: float = 0.5,         # Added parameter for noise regularization
        optim_botorch: int = 0,
        optimal_V: Optional[np.ndarray] = None,
        Q: Optional[Callable] = None,
        V: Optional[Callable] = None,
        train_freq: Union[int, tuple[int, str]] = (1, "episode"),
        logging: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self.kernel = kernel
        self.backend = backend
        #self.env = gym.make(env) if isinstance(env, str) else env
        self.env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode= "human")
        self.beta = beta
        self.len_scale= len_scale
        self.noise_reg= noise_reg
        self.horizon = horizon
        self.Q = Q
        self.V = V
        self.train_freq = train_freq
        self.logging = logging
        self.verbose = verbose
        self.seed = seed
        self.optim_botorch = optim_botorch
        self.optimal_V= optimal_V
      
        if self.logging:
           
            wandb.init(project=logging,reinit=True, settings=wandb.Settings(start_method="thread"))
               #     if self.logging:
            wandb.run.summary["noise_reg"] = self.noise_reg
            wandb.run.summary["length_scale"] = self.len_scale
            wandb.run.summary["UCB coef"] = self.beta
            wandb.run.summary["optim_botorch"] = self.optim_botorch

    def train(self, T: int):
        
        action_space = np.arange(self.env.action_space.n)      # Assuming discrete action space
    
        # Log hyperparameters
        if self.logging:
            wandb.run.summary["episode length"] = self.horizon
            wandb.run.summary["episode number "] = T
            
        # Arrays to store episode data
        all_states = []
        all_actions = []
        all_rewards = []
        Qt= [None] * (self.horizon)


        for episode in range(T):
            if self.verbose > 0:
                print(f'Episode {episode}')

          

            # Update Q-values from previous episodes
            if episode > 0:
                for h in reversed(range(len(all_states[-1]))):  # Iterate over the most recent episode's states
                    #print('h',h)
                    X_states = []
                    X_actions = []
                    y_values = []

                    for i in range(episode):  # Process all episodes together
                        if h < len(all_states[i]):  # Ensure this step exists in the episode
                            X_states.append(all_states[i][h])
                            X_actions.append(all_actions[i][h])

                            if h < len(all_states[i]) - 1:  # Next state exists
                                next_state = all_states[i][h + 1]
                                actions_batch = action_space  # Consider all actions for next state
                                states_expanded = np.repeat(next_state, len(action_space))

                                max_q_value = 0  # Default if no Q-value is available
                                if Qt[h + 1]:  # Ensure a trained model exists for h+1
                                    max_q_value = np.max(
                                        self.predict_with_gp(Qt[h + 1], states_expanded, actions_batch)[0]
                                    )

                                Qnext = max_q_value
                            else:
                                Qnext = 0  # No next state for this episode

                            y_values.append(all_rewards[i][h] + Qnext)

                    # Stack the collected data and train once for all episodes
                    if X_states:  # Ensure we have data before training
                        X = np.column_stack((X_states, X_actions))
                        y = np.array(y_values)
                        #print('X',X)
                        #print('y',y)
                        Qt[h] = self.GP_regression_torch(X, y)

                

                
                # for h in reversed(range(len(all_states[-1]))):  # Iterate over the most recent episode states
                #     for i in range(episode):  # Process each episode separately
                #         if h < len(all_states[i]):  # Ensure the step exists for this episode
                #             X_state = np.array([all_states[i][h]])
                #             X_action = np.array([all_actions[i][h]])
                #             X = np.column_stack((X_state, X_action))

                #             if h < len(all_states[i]) - 1:  # Check if a next state exists for this episode
                #                 next_state = all_states[i][h + 1]
                #                 actions_batch = action_space  # Consider all actions for next state
                #                 states_expanded = np.repeat(next_state, len(action_space))

                #                 max_q_value = 0  # Default if no Q-value is available
                #                 if Qt[h + 1]:  # Ensure a trained model exists for h+1
                #                     max_q_value = np.max(
                #                         self.predict_with_gp(Qt[h + 1], states_expanded, actions_batch)[0]
                #                     )

                #                 Qnext = max_q_value  # Get the maximum Q-value for the next state
                #             else:
                #                 Qnext = 0  # No next state for this episode

                #             y = np.array([all_rewards[i][h] + Qnext])
                #             Qt[h] = self.GP_regression_torch(X, y)
                
               

          # Execute episode
              # Initialize arrays for the current episode
            episode_states = []
            episode_actions = []
            episode_rewards = []

            initial_state = self.env.reset()
            print('initial_state',initial_state)
            state= initial_state[0] #specific to frozen lake environment
            print('state',state)
            for h in range(self.horizon):
            
                if Qt[h]:  # Ensure a model is available for the current step

                ########################################
                    states_batch = np.tile(np.arange(16), 4)  # 16 states, each repeated 4 times for each action
                    actions_batch = np.repeat(np.arange(4), 16)  # 4 actions, each repeated 16 times for each state

                    # Get posterior mean and variance for all state-action pairs
                    _, _,variances = self.predict_with_gp(Qt[h], states_batch, actions_batch)
                    
                    # Print posterior variance for each state-action tuple
                    print(f"Episode {episode}, Horizon step {h}: Posterior Variances for all state-action tuples:")
                    for i in range(16):  # Loop over states
                        for j in range(4):  # Loop over actions
                            print(f"State {i}, Action {j}: Variance {variances[i * 4 + j]}")
                ########################################
               
                    # Prepare inputs for batched prediction
                    states_batch = np.full(len(action_space), state)  # Repeat current state for all actions
                    #print('current state',states_batch)
                    actions_batch = np.array(action_space)  # Convert action_space to numpy array
                    #print('all actions',actions_batch)

                    # Predict Q-values for all actions in a single batch
                    q_values = self.predict_with_gp(Qt[h], states_batch, actions_batch)[0]
                    #print('q values',q_values)
                else:
                    # Default Q-values if no model is available
                    q_values = np.zeros(len(action_space))

                # Select action with the highest Q-value
                #self.env.render()  # Render the environment
            
                action = action_space[np.argmax(q_values)]
              

                next_state, reward, done, truncated , info = self.env.step(action)
            
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                  # If done or truncated, break the loop early
                if done or truncated:
                    break
          
                state = next_state
                #time.sleep(1.0) 
            
            
            print('len(episode_states)',len(episode_states))
            #print('episodes states after padding',len(episode_states))   
            all_states.append(np.array(episode_states))
            #print('all_states',all_states)
            all_actions.append(np.array(episode_actions))
            #print('all_actions',all_actions)
            all_rewards.append(np.array(episode_rewards))
            #print('all_rewards',all_rewards)


            episode_cum_rewards = np.sum(episode_rewards)
            episode_regret = self.optimal_V[initial_state[0]]- episode_cum_rewards #specific to frozen lake initial_state[0]
            print('episode_regret',episode_regret)
            #print('episode_cum_rewards',episode_cum_rewards)
            #initial_state_index = episode_states[0]
            #episode_regret = optimal_V[initial_state_index] - episode_cum_rewards

            if self.logging:
                metrics = {
                    "Episode_number": episode,
                    "Episode_Regret": episode_regret,
                    "Episode_Rewards": episode_cum_rewards
                }
                wandb.log(metrics)
    
    def GP_regression_torch(self, X, y):
        """
        Gaussian Process regression using PyTorch.
        
        :param X: Input tensor of shape (n_samples, n_features)
        :param y: Target tensor of shape (n_samples,)
        :return: Trained GP model
        """
         # Ensure inputs are torch tensors and use double precision
        X = torch.tensor(X, dtype=torch.float64,device=device)
        y = torch.tensor(y, dtype=torch.float64,device=device)

        # Extract number of states and actions from the environmentrb
        n_states = self.env.observation_space.n  # Total number of discrete states
        n_actions = self.env.action_space.n     # Total number of discrete actions

        # Scale states and actions to [0, 1]
        states_scaled = X[:, 0] / (n_states - 1)
        actions_scaled = X[:, 1] / (n_actions - 1)
        X_scaled = torch.stack((states_scaled, actions_scaled), dim=1).to(device)
        #print('X_scaled',X_scaled)
       

        #likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SingleTaskGP(train_X=X_scaled,train_Y= y.unsqueeze(-1).to(device)) #,outcome_transform=Standardize(m=1))  # GP expects (n_samples, 1) for targets
      
        #model.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(0.05)) #aya check this and outcome transform
        # Replace the default kernel (MaternKernel) with RBFKernel
        model.covar_module = ScaleKernel(
            RBFKernel()  # Enable ARD for different length scales per dimension
        ).to(device) 


# Set and freeze the length scale
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [self.len_scale], dtype=torch.float64, device=device
        )
        
        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        #print('len scale',model.covar_module.base_kernel.lengthscale)  # View current length scale


        # Set and freeze the noise
        model.likelihood.noise = torch.tensor([self.noise_reg], dtype=torch.float64, device=device)  # Set noise to 0.05
        if self.optim_botorch == 0:
            #print('disabled')
            model.likelihood.raw_noise.requires_grad = False

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model).to(device)
        #print('model.covar_module',model.covar_module)  # Will show ScaleKernel and RBFKernel
       

        # Inspect noise parameter
        #print('noise',model.likelihood.noise)  # Current noise value
        #print(f"Initial noise value: {model.likelihood.noise.item()}")
      
        fit_gpytorch_mll(mll)
        #print('len scale after',model.covar_module.base_kernel.lengthscale)
  

        #print(f"Optimized noise value: {model.likelihood.noise.item()}")


       
        return model
    
    def predict_with_gp(self, model, states_batch, actions_batch):

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n

        state_scaled = np.array(states_batch) / (n_states - 1)
        action_scaled = np.array(actions_batch) / (n_actions - 1)

        # Combine states and actions into a single batch
        states_expanded = state_scaled.reshape(-1, 1)  # Ensure column vector
        actions_expanded = action_scaled.reshape(-1, 1)  # Ensure column vector
        X_combined = np.hstack((states_expanded, actions_expanded))  # Shape: (batch_size, 2)
        X_combined = torch.tensor(X_combined, dtype=torch.float64, device=device)
    
      
        #print('X combined',X_combined)
        
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_combined)
            mean = posterior.mean.squeeze(-1).cpu().numpy()  # Shape: (batch_size,)
            std_dev = posterior.variance.sqrt().squeeze(-1).cpu().numpy()  # Shape: (batch_size,)
        #print('std_dev',std_dev)
        # Compute mean + beta * std_dev for each batch element
        acquisition_values = mean + self.beta * std_dev
        #print('acquisition_values',acquisition_values)
        return acquisition_values, mean, std_dev


 
# Example usage
if __name__ == "__main__":
    #env = "MiniGrid-Empty-5x5-v0"  # Replace with your desired MiniGrid environment
    env= "FrozenLake-v1"
    parser = argparse.ArgumentParser(description="Run KRVI Algorithm")
    # Adding arguments for user input
    parser.add_argument("--beta", type=float, default=0.1, help="UCB coefficient")
    parser.add_argument("--horizon", type=int, default=100, help="Horizon length")
    parser.add_argument("--len_scale", type=float, default=0.1, help="Length scale for GP kernel")
    parser.add_argument("--noise_reg", type=float, default=0.5, help="Noise regularization for GP")
    parser.add_argument("--env", type=str, default="FrozenLake-v1", help="Environment name")
    parser.add_argument("--train_freq", type=int, default=1, help="Training frequency per episode")
    parser.add_argument("--logging", type=str, default="True", help="Enable or disable logging")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info)")
    parser.add_argument("--iterations", type=int, default=500, help="Number of training iterations (T)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--optim_botorch", type= int, default = 0, help ='turn on hyperparm optimization')



    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #optimal_V= np.zeros(16)
    optimal_V= np.load('/scratch/rmapkay/kernel_based_RL_submission_modified/optimal_value_function_2000.npy')
    print('optimal_V',optimal_V)
     # Create KRVI instance with parsed arguments
    krvi = KRVI(
        kernel="RBF",
        backend="GP",
        env=args.env,
        beta=args.beta,
        horizon=args.horizon,
        len_scale=args.len_scale,
        noise_reg=args.noise_reg,
        optim_botorch=args.optim_botorch,
        optimal_V = optimal_V,
        Q=None,
        V=None,
        train_freq=(args.train_freq, "episode"),
        logging=args.logging,
        verbose=args.verbose
    )

    #krvi = KRVI(kernel="RBF", backend="GP", env=env, beta=0.1,len_scale=0.1, noise_reg=0.5, horizon=100, Q=None, V=None, train_freq=(1, "episode"),logging= 'True',  verbose=1)
    krvi.train(T= args.iterations)


