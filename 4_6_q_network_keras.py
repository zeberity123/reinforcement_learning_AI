# 4_6_q_network_keras.py
import gym
import numpy as np
import keras


def e_greedy(i, env, vector):
    e = 1 / (i // 100 + 1)
    return env.action_space.sample() if np.random.rand() < e else np.argmax(vector)


def random_noise(i, env, vector):
    return np.argmax(vector + np.random.randn(env.action_space.n) / (i + 1))


def make_onehot(state):
    z = np.zeros(16)
    z[state] = 1
    return z.reshape(1, -1)         # (1, 16)


def q_network_keras():
    env = gym.make('FrozenLake-v0')         # 미끄러운 빙판

    model = keras.Sequential([
        keras.layers.Input(shape=[16]),     # state
        keras.layers.Dense(4),              # action
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.mse)

    success, discounted, lr = 0, 0.99, 0.75
    for i in range(2000):
        state = env.reset()
        # env.render()

        done = False
        while not done:
            p = model.predict(make_onehot(state), verbose=0)    # (1, 4)
            action = random_noise(i, env, p[0])
            next_state, reward, done, _ = env.step(action)

            if done:
                p[0, action] = reward
            else:
                p_next = model.predict(make_onehot(next_state), verbose=0)
                p[0, action] = reward + discounted * np.max(p_next)

            model.fit(make_onehot(state), p, epochs=1, verbose=0)
            state = next_state

            # env.render()
        success += reward

        if i % 10 == 0:
            print(i, '성공 :', success, success / (i+1))


q_network_keras()
