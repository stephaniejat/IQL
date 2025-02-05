import numpy as np
import gymnasium as gym
np.random.seed(0)


# Create FrozenLake environment
env =  gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode= "human")

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 1.0    # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 2000 #5000  # Number of episodes

# Initialize Q-table (State x Action)
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Training loop
for episode in range(episodes):
    print('episode',episode)
    state = env.reset()[0]
    done = False

    while not done:
        # Choose action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state, :])  # Exploit

        # Take action and observe outcome
        next_state, reward, done, _, _ = env.step(action)

        # Q-learning update rule
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state  # Move to next state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Optimal Value Function of States

V = np.max(Q, axis=1)#.reshape((4, 4))  # Reshaping for 4x4 FrozenLake
print("Optimal State Value Function:\n", V)
# np.save(f'optimal_value_function.npy', V)
np.save(f'optimal_value_function_{episodes}.npy', V)


# Display final Q-table
print("\nFinal Q-table:")
print(Q)

env.close()
