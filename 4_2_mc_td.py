# 4_2_mc_td.py
import numpy as np

np.set_printoptions(precision=2)

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


# 4 x 4
class Environment:
    def __init__(self):
        self.row = 0
        self.col = 0
        self.size = 4

    def reset(self):
        self.row = 0
        self.col = 0

        return self.row, self.col

    def step(self, action):
        if action == LEFT and self.col > 0:
            self.col -= 1
        elif action == RIGHT and self.col < self.size - 1:
            self.col += 1
        elif action == UP and self.row > 0:
            self.row -= 1
        elif action == DOWN and self.row < self.size - 1:
            self.row += 1

        return (self.row, self.col), -1, self.is_done()

    def is_done(self):
        return self.row == self.size - 1 and self.col == self.size - 1


def select_action():
    coin = np.random.rand()
    return int(coin / 0.25)


def mc_method(n_iteration, alpha):
    env = Environment()
    grid = np.zeros([4, 4])
    print(grid)

    for k in range(n_iteration):
        state = env.reset()

        episode = []
        done = False
        while not done:
            action = select_action()
            state, reward, done = env.step(action)

            episode.append((state, reward))

        cum_reward = 0
        for state, reward in episode[::-1]:
            grid[state] += alpha * (cum_reward - grid[state])
            cum_reward += reward

    print(grid)


def td_method(n_iteration, alpha):
    env = Environment()
    grid = np.zeros([4, 4])
    print(grid)

    for k in range(n_iteration):
        state = env.reset()

        done = False
        while not done:
            action = select_action()
            next_state, reward, done = env.step(action)

            grid[state] += alpha * (reward + grid[next_state] - grid[state])
            # print(action, reward + grid[next_state] - grid[state])
            # print(grid, end='\n\n')
            state = next_state

    print(grid)


# mc_method(n_iteration=100, alpha=0.001)
td_method(n_iteration=100, alpha=0.1)

# for i in range(10):
#     print(select_action())

