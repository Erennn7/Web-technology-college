import numpy as np
import random

# number of slot machines (arms)
k = 5

# true reward probabilities of each arm
true_prob = [0.2, 0.5, 0.75, 0.3, 0.6]

# estimated reward values
Q = [0] * k

# number of times each arm was chosen
N = [0] * k

# exploration rate
epsilon = 0.1

# number of iterations
steps = 1000

total_reward = 0

for step in range(steps):

    # epsilon-greedy action selection
    if random.random() < epsilon:
        action = random.randint(0, k-1)   # explore
    else:
        action = np.argmax(Q)             # exploit

    # generate reward (1 or 0)
    reward = 1 if random.random() < true_prob[action] else 0

    total_reward += reward

    # update count
    N[action] += 1

    # update estimated value (incremental formula)
    Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])

print("Estimated Rewards:", Q)
print("Number of times each arm selected:", N)
print("Total Reward:", total_reward)