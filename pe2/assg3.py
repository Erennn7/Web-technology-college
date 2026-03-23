import numpy as np

grid_size = 4

actions = [(-1,0),(1,0),(0,-1),(0,1)]

reward = -1

gamma = 1.0

V = np.zeros((grid_size, grid_size))

terminal_states = [(0,0),(3,3)]

def get_next_state(state, action):
    x, y = state
    dx, dy = action
    nx, ny = x + dx, y + dy

    if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size:
        return state
    return (nx, ny)

def value_iteration():
    global V
    theta = 0.0001

    while True:
        delta = 0
        new_V = V.copy()

        for i in range(grid_size):
            for j in range(grid_size):

                if (i,j) in terminal_states:
                    continue

                values = []

                for action in actions:
                    next_state = get_next_state((i,j), action)
                    x,y = next_state
                    values.append(reward + gamma * V[x][y])

                new_V[i][j] = max(values)

                delta = max(delta, abs(new_V[i][j] - V[i][j]))

        V = new_V

        if delta < theta:
            break

def print_values():
    print("State Value Function:")
    print(np.round(V,2))

value_iteration()
print_values()

assg3