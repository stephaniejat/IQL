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
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter

from DoubleDQN import DoubleDQN
from utils import generate_random_map_with_fixed_lakes
from FrozenLakeStateWrapper import FrozenLake2DStateWrapper
from KRVI.SymFeatureExtractor import SymmetricFeatureExtractor


MAX_EP_LEN = 50
LOG_DIR = "./logs/"



DESC = generate_random_map_with_fixed_lakes(4, 4)
desc_list = [DESC]

for i in range(3):
    desc = list(zip(*DESC[::-1]))
    desc_list.append(desc)

def get_multi_vec_env():
    def fn():
        env = gym.make('FrozenLake-v1', desc=desc_list.pop(), map_name=None, is_slippery=False, max_episode_steps=MAX_EP_LEN)
        env = Monitor(FrozenLake2DStateWrapper(env, rescale = True), LOG_DIR)
        return env
    return fn

vec_env = make_vec_env(get_multi_vec_env(), n_envs = 4)

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

results_plotter.plot_results(
    [LOG_DIR], 1e6, results_plotter.X_TIMESTEPS, "FrozenLake DQN"
)