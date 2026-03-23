import numpy as np
import random

# Grid size
grid_size = 5

# Initialize rewards
rewards = np.full((grid_size, grid_size), -1)

# Define traps
traps = [(1, 2), (2, 2), (3, 1)]
for t in traps:
    rewards[t] = -10

# Define treasures
treasures = [(0, 3), (3, 3)]
for tr in treasures:
    rewards[tr] = 15

# Define goal
goal = (4, 4)
rewards[goal] = 20

# Q-table: 25 states × 4 actions
Q = np.zeros((grid_size * grid_size, 4))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 2000

# Convert position → state number
def state_number(position):
    return position[0] * grid_size + position[1]

# Take action
def take_action(position, action):
    row, col = position
    
    if action == 0 and row > 0:       # Up
        row -= 1
    elif action == 1 and row < 4:     # Down
        row += 1
    elif action == 2 and col > 0:     # Left
        col -= 1
    elif action == 3 and col < 4:     # Right
        col += 1
    
    return (row, col)

# Training
for episode in range(episodes):
    position = (0, 0)
    collected = set()   # Track collected treasures
    
    while position != goal:
        state = state_number(position)
        
        # Epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])
        
        new_position = take_action(position, action)
        reward = rewards[new_position]
        
        # Treasure collected only once
        if new_position in treasures and new_position not in collected:
            collected.add(new_position)
        elif new_position in treasures and new_position in collected:
            reward = -1  # after collected, behave as normal cell
        
        new_state = state_number(new_position)
        
        # Q update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[new_state]) - Q[state, action]
        )
        
        position = new_position

# Test best learned path
position = (0, 0)
path = [position]
total_reward = 0
visited_treasure = set()

while position != goal:
    state = state_number(position)
    action = np.argmax(Q[state])
    position = take_action(position, action)
    path.append(position)
    
    reward = rewards[position]
    
    if position in treasures and position not in visited_treasure:
        total_reward += reward
        visited_treasure.add(position)
    elif position in treasures:
        total_reward += -1
    else:
        total_reward += reward

print("Best Path Learned:")
print(path)
print("Total Reward Collected:", total_reward)