# 4_5_frozen_lake_slippery.py
import gym          # 0.21.0
import numpy as np
import util


# vector: q_table[state]
def e_greedy(i, env, vector):
    e = 1 / (i // 100 + 1)
    return env.action_space.sample() if np.random.rand() < e else np.argmax(vector)


def random_noise(i, env, vector):
    return np.argmax(vector + np.random.randn(env.action_space.n) / (i + 1))


# stochastic world, non-deterministic world
def q_learning_slippery():
    env = gym.make('FrozenLake-v1')         # 미끄러운 빙판

    # state = env.reset()
    # next_state, reward, done, info = env.step(1)
    # print(info)             # {'prob': 0.3333333333333333}

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    success, discounted, lr = 0, 0.99, 0.85
    for i in range(2000):
        state = env.reset()
        # env.render()

        done = False
        while not done:
            # action = e_greedy(i, env, q_table[state])
            action = random_noise(i, env, q_table[state])
            next_state, reward, done, _ = env.step(action)

            q_table[state, action] = ((1 - lr) * q_table[state, action] +
                                      lr * (reward + discounted * np.max(q_table[next_state])))
            state = next_state

            # env.render()
        success += reward

    util.draw_q_table(q_table, holes=[(1, 1), (1, 3), (2, 3), (3, 0)])
    print('성공 :', success, success / 2000)


q_learning_slippery()
