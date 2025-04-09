from typing import Any, Callable, Final, Sequence, Tuple

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType

from utils import generate_random_map

from typing import Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils import generate_random_map, generate_random_map_with_fixed_lakes  # Ensure this is correctly imported

class FrozenLake2DStateWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, rescale=False, size=4, p=0.8):
        super().__init__(env)
        self.rescale = rescale
        self.size = size
        self.p = p  # Probability of frozen tiles

        self.char_map = {b'S': 0, b'F': 1, b'H': 2, b'G': 3}

        self.desc = None  # Will be initialized in reset
        self.grid_size = size
        self.start_position = None
        self.goal_position = None
        self.hole_positions = None

        self._initialize_map()  # Initialize the first map

    def _initialize_map(self): #generate a random map where S and G are still restricted to opposite corners, holes position and number are random
        """Generates a new random map and updates attributes."""
        rotate= random.randint(0, 3)
        # print('rotate',rotate)
        #self.desc = np.array(generate_random_map(size=self.size, p=self.p, rotate=random.randint(0, 3)), dtype='c')
        self.desc = np.array(generate_random_map_with_fixed_lakes(size=self.size, n_hole=4, rotate=rotate), dtype='c')
        self.grid_size = len(self.desc)
        
        self.hole_positions = []
        for i, row in enumerate(self.desc):
            for j, item in enumerate(row):
                if item == b'H':
                    self.hole_positions.append([i, j])
                elif item == b'S':
                    self.start_position = np.array([i, j])
                elif item == b'G':
                    self.goal_position = np.array([i, j])

        self.hole_positions = np.array(self.hole_positions)
        # print('self.start_position',self.start_position)
        # print('self.goal_position',self.goal_position)
        # print('self.hole_positions',self.hole_positions)

        # Update observation space
        n_holes = len(self.hole_positions)
        self.observation_space = spaces.Box(
            low=-0.5 * self.grid_size + 0.5,
            high=3 - self.grid_size * 0.5 + 0.5,
            shape=(n_holes + 2, 2),
            dtype=np.float32
        )
       # ✅ Reinitialize the environment with the new map
        self.env = gym.make("FrozenLake-v1", desc=self.desc.tolist(), is_slippery=False)

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        """Resets the environment and generates a new random map."""
        self._initialize_map()  # Generate a new random map at each reset
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation: Any) -> Any:
        """Transforms the environment's observation into the 2D state representation."""
        current_pos = np.array(list(self._get_position_from_obs(observation)))
        pos_obs = np.concatenate(
            [
                current_pos.reshape(1, -1),
                self.goal_position.reshape(1, -1),
                self.hole_positions
            ]
        ) * 1.0
        if self.rescale:
            pos_obs = pos_obs - 0.5 * self.grid_size + 0.5
        return pos_obs

    def _get_position_from_obs(self, obs: int) -> Tuple[int, int]:
        """Converts observation (integer state) to grid coordinates."""
        return divmod(obs, self.grid_size)





if __name__ == "__main__":
    base_env = gym.make("FrozenLake-v1", is_slippery=False)
    print(base_env.unwrapped.desc)

    # Step 2: Wrap it with FrozenLake2DStateWrapper
    env = FrozenLake2DStateWrapper(base_env, rescale=True)
    #env = FrozenLake2DStateWrapper(None, rescale=True)  # Start with no env, let wrapper create it
    env.reset()

    # env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
    # print(env.unwrapped.desc)
    # env = FrozenLake2DStateWrapper(env, rescale = True)
    # env.reset()
    print(env.desc)
    for i in range(100):
        action = int(input())
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward, terminated, truncated, info)

