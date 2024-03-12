# 4_4_frozen_lake_discount.py
import gym          # 0.21.0
import numpy as np
import util


# vector: q_table[state]
def e_greedy(i, env, vector):
    e = 1 / (i // 100 + 1)
    return env.action_space.sample() if np.random.rand() < e else np.argmax(vector)


def random_noise(i, env, vector):
    return np.argmax(vector + np.random.randn(4) / (i + 1))


def q_learning_discount():
    id_name = "MyLake-v1"
    gym.envs.register(id_name, entry_point='gym.envs.toy_text:FrozenLakeEnv')

    env = gym.make(id_name, map_name='4x4', is_slippery=False)

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    success, discounted = 0, 0.99
    for i in range(2000):
        state = env.reset()
        # env.render()

        done = False
        while not done:
            # action = e_greedy(i, env, q_table[state])
            action = random_noise(i, env, q_table[state])
            next_state, reward, done, _ = env.step(action)

            q_table[state, action] = reward + discounted * np.max(q_table[next_state])
            state = next_state

            # env.render()
        success += reward

    util.draw_q_table(q_table, holes=[(1, 1), (1, 3), (2, 3), (3, 0)])
    print('성공 :', success, success / 2000)


q_learning_discount()


# +----------------------------------------------------------+
# +             |              |              |              |
# +    STRT     |              |              |              |
# +             |              |              |              |
# +----------------------------------------------------------+
# +             |              |              |              |
# +             |     HOLE     |              |     HOLE     |
# +             |              |     0.81     |              |
# +----------------------------------------------------------+
# +             |              |              |              |
# +             |          0.81|0.72          |     HOLE     |
# +             |    0.81      |     0.9      |              |
# +----------------------------------------------------------+
# +             |              |              |              |
# +    HOLE     |          0.9 |            1 |     GOAL     |
# +             |              |              |              |
# +----------------------------------------------------------+
















