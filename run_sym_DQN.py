import re
from matplotlib import pyplot as plt
from typing import List, Tuple

import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from gymnasium import spaces

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, window_func
from stable_baselines3.common import results_plotter

from DoubleDQN import DoubleDQN
from utils import generate_random_map_with_fixed_lakes
from FrozenLakeStateWrapper import FrozenLake2DStateWrapper
from KRVI.SymFeatureExtractor import SymmetricFeatureExtractor


MAX_EP_LEN = 50
LOG_DIR = "./logs/"

X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100

def plot_curves(
    xy_list: List[Tuple[np.ndarray, np.ndarray]], x_axis: str, title: str, figsize: Tuple[int, int] = (8, 2)
) -> None:
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    for _, (x, y) in enumerate(xy_list):
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean)
    plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()
    plt.savefig(title)

DESC = generate_random_map_with_fixed_lakes(4, 4)
desc_list = [DESC]


results_plotter.plot_curves = plot_curves


for i in range(3):
    desc = list(zip(*DESC[::-1]))
    desc_list.append(desc)

def get_multi_vec_env():
    def fn():
        env = gym.make('FrozenLake-v1', desc=desc_list.pop(), map_name=None, is_slippery=False, max_episode_steps=MAX_EP_LEN)
        env = FrozenLake2DStateWrapper(env, rescale = True)
        return env
    return fn

vec_env = make_vec_env(get_multi_vec_env(), n_envs = 4)
vec_env = VecMonitor(vec_env, LOG_DIR)

eval_env=gym.make('FrozenLake-v1', desc=DESC, map_name=None, is_slippery=False, max_episode_steps=MAX_EP_LEN)
eval_env = FrozenLake2DStateWrapper(eval_env, rescale = True)

# TODO: add break
# TODO: Max_episode len

policy_kwargs = dict(
    features_extractor_class=SymmetricFeatureExtractor,
    features_extractor_kwargs=dict(out_features_dim=4),
    net_arch= [128, 64, 32]
)

model = DQN(
    "MlpPolicy", 
    vec_env, 
    verbose=1,
    # n_steps = 2,
    batch_size = 256,
    train_freq = (4, "step"),
    gradient_steps = 2,
    learning_starts = 500,
    exploration_initial_eps = 1,
    exploration_final_eps = 0.05,
    learning_rate = 0.0001,
    # tau = 0.9,
    # policy_kwargs = {"net_arch": [128, 64, 32]}
    policy_kwargs = policy_kwargs,
    target_update_interval = 500
    )

# print(model.policy)


eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=1000,
                             deterministic=True, render=False)

try:
    model.learn(total_timesteps=1000000, log_interval=500, callback=eval_callback)
except KeyboardInterrupt:
    print("Interrupted")
    pass
# model.save("dqn_frozenlake_multi")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_frozenlake_multi")

# obs, info = eval_env.reset()
# i = 0
# while True:
#     i+=1
#     action, _states = model.predict(obs, deterministic=True)
#     action = action.item()
#     obs, reward, terminated, truncated, info = eval_env.step(action)
#     if terminated:
#         print("Reward in attempt {} is {}. YAYAYA!".format(i, reward))
#     if terminated or truncated:
#         obs, info = eval_env.reset()

mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    deterministic=False,
    n_eval_episodes=64,
)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


  

def clean_csv(file_path):
    # Define a regular expression pattern to match valid rows
    pattern = re.compile(r'^\d+\.\d+,\d+,\d+\.\d+$')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the header
    header = lines[1]
    
    # Filter out invalid rows
    valid_lines = [header]
    for line in lines[2:]:
        if pattern.match(line.strip()):
            valid_lines.append(line)
    
    # Write the cleaned data back to the file
    with open(file_path, 'w') as file:
        file.write(lines[0])  # Write the first line (metadata)
        file.writelines(valid_lines)

# clean_csv(LOG_DIR + 'monitor.csv')



results_plotter.plot_results(
    [LOG_DIR], 1e6, results_plotter.X_TIMESTEPS, "FrozenLake DQN"
)       