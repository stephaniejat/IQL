from typing import Any, Callable, Final, Sequence, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType

from utils import generate_random_map

class FrozenLake2DStateWrapper(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        rescale = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, rescale=rescale,
        )
        gym.ObservationWrapper.__init__(self, env)

        self.rescale = rescale

        self.desc = env.unwrapped.desc
        self.char_map = {
            b'S': 0,
            b'F': 1,
            b'H': 2,
            b'G': 3
        }
        self.grid_size = len(self.desc)

        self.hole_positions = []
        for i, row in enumerate(self.desc):
            for j, item in enumerate(row):
                if item == b'F':
                    continue
                if item == b'H':
                    self.hole_positions.append([i,j])
                    continue
                if item == b'S':
                    self.start_position = np.array([i,j])
                    continue
                if item == b'G':
                    self.goal_position = np.array([i,j])

        n_holes = len(self.hole_positions)
        self.hole_positions = np.array(self.hole_positions)

        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(n_holes+3, 2), dtype=int)


    def observation(self, observation: ObsType) -> Any:
        # Observation space is a concatenation of 2D positional vectors:
        # 1. Current position
        # 2. Starting position 
        # 3. Goal position
        # 4. Hole position
        current_pos = np.array(list(self._get_position_from_obs(observation)))
        pos_obs = np.concatenate(
            [
                current_pos.reshape(1,-1), 
                self.start_position.reshape(1,-1), 
                self.goal_position.reshape(1,-1), 
                self. hole_positions
                ]
                ) * 1.0
        if self.rescale:
            pos_obs -= self.grid_size * 0.5
        return pos_obs

    def _get_position_from_obs(self, obs: int) -> Tuple[int]:
        return divmod(obs, self.grid_size)


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
    print(env.unwrapped.desc)
    env = FrozenLake2DStateWrapper(env, rescale = True)
    env.reset()
    for i in range(100):
        action = int(input())
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward, terminated, truncated, info)