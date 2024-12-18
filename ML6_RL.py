import numpy as np
import random

# Define the maze environment
maze = np.array([
    [0, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
])

# Parameters
n_rows, n_cols = maze.shape
n_states = n_rows * n_cols
n_actions = 4
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.8
n_episodes = 1000

# Get start and goal states
start_state = int(input("Enter start state: "))
goal_state = int(input("Enter goal state: "))

# Initialize Q-table
q_table = np.zeros((n_states, n_actions))

# Define actions
actions = ["up", "down", "left", "right"]

# Reward function
def reward(state):
    return 10 if state == goal_state else -1

# Determine next state based on action
def next_state(state, action):
    row, col = divmod(state, n_cols)
    if action == 0 and row > 0 and maze[row - 1, col] == 0:  # Up
        row -= 1
    elif action == 1 and row < n_rows - 1 and maze[row + 1, col] == 0:  # Down
        row += 1
    elif action == 2 and col > 0 and maze[row, col - 1] == 0:  # Left
        col -= 1
    elif action == 3 and col < n_cols - 1 and maze[row, col + 1] == 0:  # Right
        col += 1
    return row * n_cols + col

# Q-learning algorithm
for _ in range(n_episodes):
    state = start_state
    while state != goal_state:
        action = random.randint(0, n_actions - 1) if random.uniform(0, 1) < epsilon else np.argmax(q_table[state])
        new_state = next_state(state, action)
        r = reward(new_state)
        q_table[state, action] += learning_rate * (r + discount_factor * np.max(q_table[new_state]) - q_table[state, action])
        state = new_state

# Get the optimal path
def get_optimal_path():
    state = start_state
    path = [state]
    while state != goal_state:
        action = np.argmax(q_table[state])
        state = next_state(state, action)
        path.append(state)
    return path


# Display the final Q-table
print("Trained Q-Table:")
print(q_table)

optimal_path = get_optimal_path()
print("Optimal path:", optimal_path)
