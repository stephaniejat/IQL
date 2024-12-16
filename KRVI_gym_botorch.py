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
        self.env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
        self.beta = beta
        self.horizon = horizon
        self.Q = Q
        self.V = V
        self.train_freq = train_freq
        self.logging = logging
        self.verbose = verbose
        self.seed = seed
      
        if logging:
           
            wandb.init(project=logging)

    def train(self, T: int):
        print('self.env',self.env)
        #self.env = FullyObsWrapper(self.env) #specific to minigrid
        print('state space',self.env.observation_space)
        #print('state space after wrapping',self.env.observation_space['image'].sample())

        #state_space = np.arange(self.env.observation_space.n)  # Assuming discrete observation space
        print('action space', self.env.action_space)
        action_space = np.arange(self.env.action_space.n)      # Assuming discrete action space
        #print('action space after',action_space)
        #state_action_space = [(s, a) for s in state_space for a in action_space]

        #optimal_V = np.zeros(len(state_space))  # Placeholder for optimal value function for continuous state

        # Log hyperparameters
        if self.logging:
            wandb.run.summary["episode length"] = self.horizon
            wandb.run.summary["episode number"] = T
            wandb.run.summary["UCB coef"] = self.beta

        # Arrays to store episode data
        all_states = []
        all_actions = []
        all_rewards = []
        Qt= [None] * (self.horizon + 1)

        #Qt = np.zeros((self.horizon, len(state_action_space)))
        #Qt_std = np.zeros((self.horizon, len(state_action_space)))
        #Qt_estimate = np.zeros((self.horizon + 1, len(state_space), len(action_space)))

        for episode in range(T):
            if self.verbose > 0:
                print(f'Episode {episode}')

            # Initialize arrays for the current episode
            episode_states = []
            episode_actions = []
            episode_rewards = []

            state = self.env.reset()
            state= state[0] #specific to frozen lake environment
            print('initial state',state)

            # Update Q-values from previous episodes
            if episode > 0:
                #print('episode',episode)
                for h in reversed(range(self.horizon)):
                    print('h',h)
                   
                    if episode == 1:
                        # If it's the first episode, initialize X_states and X_actions directly
                        X_states = np.array([all_states[0][h]])  # Use np.array to create a 1D array
                        X_actions = np.array([all_actions[0][h]])  # Likewise for actions
                    else:
                        # For subsequent episodes, concatenate as usual
                        X_states = np.concatenate([np.array([all_states[i][h]]) for i in range(episode)])

                        X_actions = np.concatenate([np.array([all_actions[i][h]]) for i in range(episode)])
                    #X_states = np.concatenate([all_states[i][h] for i in range(episode)])
                    #X_actions = np.concatenate([all_actions[i][h] for i in range(episode)])
                    X = np.column_stack((X_states, X_actions))
                    print('X stacked numpy',X)
                    #X_torch = torch.tensor(X, dtype=torch.float64)
                    #print('X torch',X_torch)

                    

                    #X_states = X_states.reshape(-1, 1)
                    #X_actions = X_actions.reshape(-1, 1)

                    #X = np.concatenate((X_states, X_actions), axis=1)
                   

                    Qnext = []

                    if h < self.horizon - 1:
                        # Collect all next states at step h+1 for all episodes
                        next_states_batch = np.array([all_states[i][h + 1] for i in range(episode)])
                        print('next_states_batch', next_states_batch)

                        # Expand batch for all actions
                        batch_size = next_states_batch.shape[0] #which is the number of episodes so far
                        print('batch size',batch_size)
                        actions_batch = np.tile(action_space, batch_size)  # Repeat action_space for each state
                        print('actions_batch',actions_batch)
                        states_expanded = np.repeat(next_states_batch, len(action_space))  # Repeat each state for all actions
                        print('states_expanded',states_expanded)

                        # Predict Q-values for all (state, action) pairs
                        max_q_values = np.zeros(batch_size)
                        if Qt[h + 1]:
                            max_q_values = np.max(
                                self.predict_with_gp(Qt[h + 1], states_expanded, actions_batch)
                                .reshape(batch_size, len(action_space)), 
                                axis=1
                            )

                        Qnext.extend(max_q_values)
                    else:
                        Qnext.extend([0] * episode)
                    print('Qnext',Qnext)

                    # for i in range(episode):
                    #     if h < self.horizon - 1:
                    #         next_state = all_states[i][h + 1]
                    #         print('next_state',next_state)
                           
                    #         next_q_values = []
                    #         for a in action_space:
                    #             # Use the prediction method to calculate mean + beta * std_dev
                    #             model = Qt[h + 1]
                    #             q_value = self.predict_with_gp(model, next_state, a)
                    #             next_q_values.append(q_value)

                    #         # Take the maximum Q-value over all actions
                    #         Qnext.append(max(next_q_values))                      
            
                    #     else:
                    #         Qnext.append(0)

                    y = np.array([all_rewards[i][h] + Qnext[i] for i in range(len(Qnext))])
                      # Convert targets to tensor
                    #y_torch = torch.tensor(y, dtype=torch.float64)
                   

                    #print('y_torch',y_torch)
                          # Perform GP regression using PyTorch
                    Qt[h] = self.GP_regression_torch(X, y)
                    print('Qt[h]',Qt[h])

                    #Qt[h], Qt_std[h] = self.GP_regression_with_RBF(X, y, state_action_space)
                   
                    #Qt_estimate[h] = Qt_reshaped + self.beta * Qt_std_reshaped


          # Execute episode
            for h in range(self.horizon):
                #state_index = state  # Direct mapping assuming discrete state space

                # Use "image" component of state space
                # q_values = [
                #     self.predict_with_gp(Qt[h], state, action) 
                #     for action in action_space
                # ] if Qt[h] else np.zeros(len(action_space))
                if Qt[h]:  # Ensure a model is available for the current step
                    # Prepare inputs for batched prediction
                    states_batch = np.full(len(action_space), state)  # Repeat current state for all actions
                    actions_batch = np.array(action_space)  # Convert action_space to numpy array

                    # Predict Q-values for all actions in a single batch
                    q_values = self.predict_with_gp(Qt[h], states_batch, actions_batch)
                else:
                    # Default Q-values if no model is available
                    q_values = np.zeros(len(action_space))

                # Select action with the highest Q-value
                action = action_space[np.argmax(q_values)]
              

                #action_index = np.argmax(q_values)
                #action_index=np.random.randint(0,3)
                #action = action_space[action_index]
                print('action',action)
                #q_values = Qt_estimate[h][state_index]  # you should get the estimate from calling gp regression for this state and all actions
                #action_index = np.argmax(q_values)

               
                result = self.env.step(action)
                print('result',result)
                next_state, reward, done, truncated , info = self.env.step(action)
                print('state',state)
                print('next state',next_state)
                episode_states.append(state)
                print('episode_states',episode_states)
                episode_actions.append(action)
                episode_rewards.append(reward)

                # if done:
                #     break
                state = next_state
            all_states.append(np.array(episode_states))
            print('all_states',all_states)
            all_actions.append(np.array(episode_actions))
            print('all_actions',all_actions)
            all_rewards.append(np.array(episode_rewards))
            print('all_rewards',all_rewards)


            episode_cum_rewards = np.sum(episode_rewards)
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
        X = torch.tensor(X, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)

        # Extract number of states and actions from the environment
        n_states = self.env.observation_space.n  # Total number of discrete states
        n_actions = self.env.action_space.n     # Total number of discrete actions

        # Scale states and actions to [0, 1]
        states_scaled = X[:, 0] / (n_states - 1)
        actions_scaled = X[:, 1] / (n_actions - 1)
        X_scaled = torch.stack((states_scaled, actions_scaled), dim=1)
        print('X_scaled',X_scaled)
       

        #likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SingleTaskGP(train_X=X_scaled,train_Y= y.unsqueeze(-1),outcome_transform=Standardize(m=1))  # GP expects (n_samples, 1) for targets
        #model.likelihood.noise_covar.noise = torch.tensor(1e-10)  # Set it to a slightly larger value
        model.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(0.05)) #aya check this and outcome transform

        #model.likelihood.noise_covar.raw_noise_constraint = GreaterThan(1e-3)


        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        #model.likelihood.noise_covar.noise = torch.tensor(1e-10)
        fit_gpytorch_mll(mll)

       
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
        X_combined = torch.tensor(X_combined, dtype=torch.float64)
    
      
        print('X combined',X_combined)
        
        
        # Assuming state_scaled and action_scaled are numpy arrays
        #X_scaled = torch.tensor(np.array([state_scaled, action_scaled]).reshape(1, -1), dtype=torch.float64)
       
        #X_scaled = torch.tensor([[state_scaled, action_scaled]], dtype=torch.float64)

        # Make predictions
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_combined)
            mean = posterior.mean.squeeze(-1).numpy()  # Shape: (batch_size,)
            std_dev = posterior.variance.sqrt().squeeze(-1).numpy()  # Shape: (batch_size,)

        # Compute mean + beta * std_dev for each batch element
        acquisition_values = mean + self.beta * std_dev
        print('acquisition_values',acquisition_values)
        return acquisition_values

    # def predict_with_gp(self, model, state, action):
   
    #     n_states = self.env.observation_space.n
    #     n_actions = self.env.action_space.n

    #     # Scale state and action to [0, 1]
    #     state_scaled = state / (n_states - 1)
    #     action_scaled = action / (n_actions - 1)
    #     X_scaled = torch.tensor([[state_scaled, action_scaled]], dtype=torch.float64)

    #     # Make predictions
    #     model.eval()
    #     with torch.no_grad():
    #         posterior = model.posterior(X_scaled)
    #         mean = posterior.mean.item()
    #         std_dev = posterior.variance.sqrt().item()

    #     # Return mean + beta * std_dev
    #     return mean + self.beta * std_dev

        

    def GP_regression_with_RBF(self, X, y, l=1, alpha=1e-10):
        # Define the RBF kernel
        kernel = RBF(length_scale=l, length_scale_bounds="fixed")

        if self.logging:
            wandb.run.summary["kernel type"] = str(kernel)
            wandb.run.summary["alpha_gp"] = alpha
            wandb.run.summary["length_scale"] = l

        # Create GPR model
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=alpha)  # Disabling kernel parameter optimization

        # Fit the model
        gpr.fit(X, y)

        # Predict mean and standard deviation
        #y_pred_mean, y_pred_std = gpr.predict(state_action_space, return_std=True)

        return gpr

    # def scale_state_action(self, states, actions):
   
    # # State scaling
    #     state_low = torch.tensor(self.env.observation_space.low, dtype=states.dtype)
    #     state_high = torch.tensor(self.env.observation_space.high, dtype=states.dtype)
    #     state_range = state_high - state_low
    #     states_scaled = (states - state_low) / state_range

    #     # Action scaling
    #     if isinstance(self.env.action_space, gym.spaces.Discrete):
    #         action_low = torch.tensor(0.0, dtype=actions.dtype)
    #         action_high = torch.tensor(self.env.action_space.n - 1, dtype=actions.dtype)
    #     else:  # Assume continuous action space
    #         action_low = torch.tensor(self.env.action_space.low, dtype=actions.dtype)
    #         action_high = torch.tensor(self.env.action_space.high, dtype=actions.dtype)
    #     action_range = action_high - action_low
    #     actions_scaled = (actions - action_low) / action_range

    #     # Combine scaled states and actions
    #     sa_scaled = torch.cat((states_scaled, actions_scaled), dim=-1)
    #     return sa_scaled, (state_low, state_high, action_low, action_high)


    # def GP_regression_torch(self,X,y):
    #     model = SingleTaskGP(X,y) #default RBF kernel)
    #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    #     fit_gpytorch_mll(mll)
 
# Example usage
if __name__ == "__main__":
    #env = "MiniGrid-Empty-5x5-v0"  # Replace with your desired MiniGrid environment
    env= "FrozenLake-v1"
    krvi = KRVI(kernel="RBF", backend="GP", env=env, beta=1.0, horizon=5, Q=None, V=None, train_freq=(1, "episode"), verbose=1)
    krvi.train(T=5)


