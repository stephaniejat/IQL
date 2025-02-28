import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import cycle

class FrozenLake2DStateWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, rescale=False):
        super().__init__(env)
        self.rescale = rescale
        self.original_desc = env.unwrapped.desc  # Store original map
        self.grid_size = len(self.original_desc)

       


        # Rotation matrices (CLOCKWISE)
        self.rotation_matrices = [
            np.eye(2),  # 0 degrees (identity)
        np.array([[0, 1], [-1, 0]]),  # 90 degrees CW
        np.array([[-1, 0], [0, -1]]),  # 180 degrees CW
        np.array([[0, -1], [1, 0]])  # 270 degrees CW
        ]

        self.rotation_cycle = cycle([0, 1, 2, 3])  # Cycle through 4 rotations
        #self.current_rotation = 1
        self.current_rotation = next(self.rotation_cycle)

        self.reset_map_and_positions()  # Initialize with rotated map

        # Define observation space
        n_holes = len(self.hole_positions)
        # change the observation space based on scaling
        self.observation_space = spaces.Box( 
            low=-0.5 * self.grid_size + 0.5, 
            high=3 - self.grid_size * 0.5 + 0.5, 
            shape=(n_holes + 2, 2), 
            dtype=np.float32
        )

    def reset_map_and_positions(self):
        """Rotates the map and updates all positions accordingly."""
        self.desc = self.rotate_map(self.original_desc, self.current_rotation)
        self.goal_position = np.array([
    (i, j) for i, row in enumerate(self.desc)
    for j, item in enumerate(row)
    if item == b'G'
])[0]  # There is only one goal, so take the first match

        self.hole_positions = np.array([
    (i, j) for i, row in enumerate(self.desc)
    for j, item in enumerate(row)
    if item == b'H'
])

        # self.goal_position = np.argwhere(self.desc == b'G')[0]
        # self.hole_positions = np.argwhere(self.desc == b'H')

    def rotate_map(self, desc, rotation_idx):
        """Applies a 90-degree rotation to the map `rotation_idx` times."""
        rotated_desc = np.array(desc)
        for _ in range(rotation_idx):
            rotated_desc = np.array(list(zip(*rotated_desc[::-1])))
        return rotated_desc

    def rotate_point(self, point, rotation_idx):
        #print('rotation_idx',rotation_idx)
        #print('rotation_matrices[rotation_idx]',self.rotation_matrices[rotation_idx])
        #print('point',point)
        """Rotates a coordinate based on rotation index."""
        return np.dot(self.rotation_matrices[rotation_idx], point)

    def rotate_action(self, action, rotation_idx): #This should be anticlowise because it is rotating actions back to the original unrotated space
        action_map = [
             [0, 1, 2, 3],  # 0°: [LEFT, DOWN, RIGHT, UP]
             [1, 2, 3, 0],  # 90°: [DOWN, RIGHT, UP, LEFT]
             [2, 3, 0, 1],  # 180°: [RIGHT, UP, LEFT, DOWN]
             [3, 0, 1, 2],  # 270°: [UP, LEFT, DOWN, RIGHT]
         ]
        return action_map[rotation_idx][action]
        



    def step(self, action):
        """Transforms the action based on the current rotation and steps."""
        rotated_action = self.rotate_action(action, self.current_rotation)
        #print('back rotated action', rotated_action)
        obs, reward, terminated, truncated, info = self.env.step(rotated_action) #step is always performed in the unrotated env
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment and samples a random rotation."""
        self.current_rotation = np.random.choice([0, 1, 2, 3])  # Random rotation
        #self.current_rotation = 1
        self.reset_map_and_positions()  # Apply new rotation
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
    #Computes the agent’s current position in the rotated grid."""
        #print('obs',observation)
        # Get the current position (row, col) in the unrotated grid
        current_pos = np.array(divmod(observation, self.grid_size))  # (row, col)
        
        
        # Rescale the position to match the scaled coordinate space
        rescaled_pos = current_pos - 0.5 * self.grid_size + 0.5  # Centering the position
        #print('rescaled',rescaled_pos)
        # Rotate the position based on the current rotation (no shift needed here, rotation happens on rescaled position)
        rotated_pos = self.rotate_point(rescaled_pos, self.current_rotation)  # Apply rotation to the rescaled position
        #print('rotated_pos',rotated_pos)

        # Create the observation by concatenating the rotated position, goal position, and hole positions
        pos_obs = np.concatenate(
            [
                rotated_pos.reshape(1, -1),  # Current agent position in rotated grid (no further rescaling needed)
                self.goal_position.reshape(1, -1),  # Goal position (needs rotation and rescaling)
                self.hole_positions,  # Hole positions (needs rotation and rescaling)
            ]
        ).astype(np.float32)

        if self.rescale:
            # If rescaling is enabled, apply rescaling only to goal and hole positions, not the agent's position because it is already rescaled and rotated
            pos_obs[1:] = pos_obs[1:] - 0.5 * self.grid_size + 0.5

        return pos_obs




# TESTING THE WRAPPER
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
    env = FrozenLake2DStateWrapper(env, rescale=True)

    for episode in range(1):
        obs, info = env.reset()
        print('initial_obs',obs)
        print(f"Episode {episode + 1} - Rotation: {env.current_rotation * 90} degrees")
        print(env.desc)
        for _ in range(5):
            #action = np.random.choice([0, 1, 2, 3])  # Random action
            action = 0
            print('action',action)
            obs, reward, done, truncated, info = env.step(action)
            print(obs, reward, done, truncated, info)
            if done:
                break

