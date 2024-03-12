# 4_3_frozen_lake_q_learning.py
import gym          # 0.21.0
import numpy as np
import util


# vector: q_table[state]
def random_argmax(vector):
    m = np.max(vector)                      # 1 0 1 0
    # k = (vector == m)                     # [ True False  True False]
    # n = np.nonzero(k)                     # (array([0, 2], dtype=int64),)

    indices = np.nonzero(vector == m)[0]    # array([0, 2], dtype=int64)

    return np.random.choice(indices)


def simulation():
    id_name = "MyLake-v1"
    gym.envs.register(id_name, entry_point='gym.envs.toy_text:FrozenLakeEnv')

    env = gym.make(id_name, map_name='4x4', is_slippery=False)

    q_table = np.zeros((16, 4))

    for i in range(3):
        state = env.reset()
        for action in [2, 2, 1, 1, 1, 2]:
            next_state, reward, done, _ = env.step(action)

            q_table[state, action] = reward + np.max(q_table[next_state])
            print(q_table)
            state = next_state

            # env.render()

    util.draw_q_table(q_table, holes=[(1, 1), (1, 3), (2, 3), (3, 0)])


def q_learning():
    id_name = "MyLake-v1"
    gym.envs.register(id_name, entry_point='gym.envs.toy_text:FrozenLakeEnv')

    env = gym.make(id_name, map_name='4x4', is_slippery=False)

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    success = 0
    for i in range(2000):
        state = env.reset()
        # env.render()

        done = False
        while not done:
            action = random_argmax(q_table[state])
            next_state, reward, done, _ = env.step(action)

            q_table[state, action] = reward + np.max(q_table[next_state])
            state = next_state

            # env.render()
        success += reward

    util.draw_q_table(q_table, holes=[(1, 1), (1, 3), (2, 3), (3, 0)])
    print('성공 :', success, success / 2000)


# simulation()
q_learning()

# random_argmax(np.array([1, 0, 1, 0]))
