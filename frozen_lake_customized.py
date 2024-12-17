import numpy as np
import gym
from gym import spaces

class CustomFrozenLakeEnv(gym.Env):
    def __init__(self):
        super(CustomFrozenLakeEnv, self).__init__()

        # Define the grid size
        self.grid_size = 4
        
        # Define the observation space (16D vector)
        self.observation_space = spaces.Box(low=0, high=3, shape=(16,), dtype=np.int32)

        # Define the action space as discrete (for internal logic)
        self.internal_action_space = spaces.Discrete(5)  # 4 movement actions + 1 "do nothing"

        # Externally, actions are represented as one-hot vectors
        self.action_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.int32)

        # Internal state variables
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.agent_pos = None
        self.goal_pos = None
        self.holes = None
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        """Reset the environment for a new episode."""
        # Clear the grid
        self.grid.fill(1)  # Fill the grid with frozen cells

        # Randomly place the agent and goal in the corners
        corners = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.agent_pos, self.goal_pos = np.random.choice(len(corners), 2, replace=False)
        self.agent_pos = corners[self.agent_pos]
        self.goal_pos = corners[self.goal_pos]

        # Place the holes in random positions, avoiding the agent and goal
        self.holes = set()
        while len(self.holes) < 4:  # Example: 4 random holes
            hole_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if hole_pos not in self.holes and hole_pos != self.agent_pos and hole_pos != self.goal_pos:
                self.holes.add(hole_pos)

        # Update the grid representation
        for r, c in self.holes:
            self.grid[r, c] = 2  # Holes
        self.grid[self.goal_pos] = 3  # Goal
        self.grid[self.agent_pos] = 0  # Agent

        # Reset step counter
        self.current_step = 0

        # Return the flattened grid as the initial observation
        return self.grid.flatten()

    def step(self, action):
        """Take an action in the environment."""
        # Convert one-hot action vector to a discrete action index
        action = np.argmax(action)

        # Check for "do nothing" action
        if action == 4:  # "Do nothing"
            self.current_step += 1
            return self.grid.flatten(), 0, self.current_step >= self.max_steps, {}

        # Define movement offsets for actions: right, down, left, up
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Compute new agent position if not "do nothing"
        new_pos = (self.agent_pos[0] + moves[action][0], self.agent_pos[1] + moves[action][1])

        # Check if the move is within bounds
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            self.agent_pos = new_pos

        # Update the grid representation
        self.grid.fill(1)  # Reset to frozen cells
        for r, c in self.holes:
            self.grid[r, c] = 2  # Holes
        self.grid[self.goal_pos] = 3  # Goal
        self.grid[self.agent_pos] = 0  # Agent

        # Check the terminal state and reward
        self.current_step += 1
        if self.agent_pos in self.holes:
            return self.grid.flatten(), 0, True, {}  # Fell in a hole
        elif self.agent_pos == self.goal_pos:
            return self.grid.flatten(), 1, True, {}  # Reached the goal
        elif self.current_step >= self.max_steps:
            return self.grid.flatten(), 0, True, {}  # Max steps reached
        else:
            return self.grid.flatten(), 0, False, {}  # Continue the episode

    def render(self, mode='human'):
        """Render the environment in a human-readable format."""
        symbols = {0: 'A', 1: '.', 2: 'H', 3: 'G'}  # Agent, Frozen, Hole, Goal
        for row in self.grid:
            print(' '.join(symbols[cell] for cell in row))
        print()

    def sample_action(self):
        """Sample a random one-hot action."""
        discrete_action = self.internal_action_space.sample()
        one_hot_action = np.zeros(5, dtype=np.int32)
        one_hot_action[discrete_action] = 1
        return one_hot_action

    def pad_episode(self, transitions):
        """Pad an episode to the maximum number of steps by adding 'do nothing' actions."""
        while len(transitions) < self.max_steps:
            last_obs, _, _, _ = transitions[-1]  # Get the last observation
            do_nothing_action = np.zeros(5, dtype=np.int32)
            do_nothing_action[4] = 1  # "Do nothing" action
            transitions.append((last_obs, do_nothing_action, 0, True))
        return transitions