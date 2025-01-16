import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from gymnasium import spaces
from stable_baselines3 import DQN


env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

# model = DQN(
#     "MlpPolicy", 
#     env, 
#     verbose=1,
#     learning_starts = 100,
#     exploration_initial_eps = 1,
#     exploration_final_eps = 0.05,
#     # target_update_interval = 100
#     )
# model.learn(total_timesteps=50000, log_interval=4)
# model.save("dqn_frozenlake")

# del model # remove to demonstrate saving and loading

model = DQN.load("dqn_frozenlake")

obs, info = env.reset()
i = 0
while True:
    i+=1
    action, _states = model.predict(obs, deterministic=False)
    action = action.item()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("Reward in attempt {} is {}. YAYAYA!".format(i, reward))
    if terminated or truncated:
        obs, info = env.reset()