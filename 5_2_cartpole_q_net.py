# 5_2_cartpole_q_net.py
import gym
import numpy as np
import keras


# 퀴즈
# frozen lake 케라스 버전을 활용해서
# cartpole에 대해 동작하는 모델을 구성하세요
def e_greedy(i, env, vector):
    e = 1 / (i // 100 + 1)
    return env.action_space.sample() if np.random.rand() < e else np.argmax(vector)


def random_noise(i, env, vector):
    return np.argmax(vector + np.random.randn(env.action_space.n) / (i + 1))


def q_network_keras():
    env = gym.make('CartPole-v1')

    model = keras.Sequential([
        keras.layers.Input(shape=[4]),                  # state
        keras.layers.Dense(2),                          # action
        # keras.layers.Dense(1, activation='sigmoid'),  # action
    ])

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse)

    result, discounted = [], 0.99
    for i in range(2000):
        state = env.reset()
        # env.render()

        done, step_count = False, 0
        while not done:
            p = model.predict(np.reshape(state, [1, -1]), verbose=0)    # (1, 4)
            action = random_noise(i, env, p[0])
            next_state, reward, done, _ = env.step(action)
            step_count += 1

            if done:
                p[0, action] = -100
            else:
                p_next = model.predict(np.reshape(next_state, [1, -1]), verbose=0)
                p[0, action] = reward + discounted * np.max(p_next)

            model.fit(np.reshape(state, [1, -1]), p, epochs=1, verbose=0)
            state = next_state

            # env.render()
        result.append(step_count)
        print(i, '횟수 :', step_count)


# 퀴즈
# 앞에서 만든 모델을 로지스틱 리그레션 버전으로 수정하세요
def q_network_keras_sigmoid():
    env = gym.make('CartPole-v1')

    model = keras.Sequential([
        keras.layers.Input(shape=[4]),                  # state
        keras.layers.Dense(1, activation='sigmoid'),  # action
    ])

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy)

    result, discounted = [], 0.99
    for i in range(2000):
        state = env.reset()
        # env.render()

        done, step_count = False, 0
        while not done:
            p = model.predict(np.reshape(state, [1, -1]), verbose=0)    # (1, 4)
            # print(p)    # [[0.50461245]]
            action = 0 if p[0, 0] < 0.5 else 1
            next_state, reward, done, _ = env.step(action)
            step_count += 1

            if done:
                p[0, 0] = -100
            else:
                p_next = model.predict(np.reshape(next_state, [1, -1]), verbose=0)
                p[0, 0] = reward + discounted * p_next[0, 0]   # np.max(p_next)

            model.fit(np.reshape(state, [1, -1]), p, epochs=1, verbose=0)
            state = next_state

            env.render()
        result.append(step_count)
        print(i, '횟수 :', step_count)


# q_network_keras()
q_network_keras_sigmoid()
