import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from collections import deque
import keras


# 2013 버전과 동일
def make_model(input_size, output_size):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([input_size]))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(output_size))

    model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.mse)
    return model


# 2013 버전과 동일
def annealing_epsilon(episode, min_e, max_e, target_episode):
    # episode가 50을 넘으면 0 반환. min_e : 0, max_e : 1
    slope = (min_e - max_e) / target_episode        # -0.02 = -1 / 50
    return max(min_e, slope * episode + max_e)      # max(0, -0.02 * epi + 1)


# 2015에 새롭게 추가한 코드
def copy_weights(model_main, model_copy):
    w1 = model_main.get_layer(index=0).get_weights()
    w2 = model_main.get_layer(index=1).get_weights()

    model_copy.get_layer(index=0).set_weights(w1)
    model_copy.get_layer(index=1).set_weights(w2)


def dqn_2015(model_main, model_copy):
    replay_buffer = deque(maxlen=20000)
    rewards_100 = deque(maxlen=100)
    history = []

    max_episodes = 500
    for episode in range(max_episodes):
        state = env.reset()

        # e는 0.02씩 줄어들다가 50번째에서 0이 되고 바뀌지 않는다. 500 / 10
        e = annealing_epsilon(episode, 0.0, 1.0, max_episodes / 10)

        done, step_count, loss = False, 0, 0
        while not done:
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(model_main.predict(np.reshape(state, [1, -1]), verbose=0))

            next_state, reward, done, _ = env.step(action)

            if done:
                reward = -1

            replay_buffer.append((state, next_state, action, reward, done))

            state = next_state
            step_count += 1

            if len(replay_buffer) > 64:
                minibatch = random.sample(replay_buffer, 64)

                states = np.vstack([x[0] for x in minibatch])
                next_states = np.vstack([x[1] for x in minibatch])
                actions = np.array([x[2] for x in minibatch])
                rewards = np.array([x[3] for x in minibatch])
                dones = np.array([x[4] for x in minibatch])

                xx = states
                yy = model_main.predict(xx, verbose=0)

                # ------------------------------------------ #
                # 유일하게 달라진 코드
                next_rewards = model_copy.predict(next_states, verbose=0)
                # ------------------------------------------ #

                yy[range(len(xx)), actions] = rewards + 0.99 * np.max(next_rewards, axis=1) * ~dones

                model_main.fit(xx, yy, batch_size=16, verbose=0)

            # 리플레이 버퍼 복사 (추가된 코드)
            if step_count % 10 == 0:
                copy_weights(model_main, model_copy)

        # 추가할 때마다 가장 오래된 것을 자동 삭제. e는 출력할 필요없다.
        rewards_100.append(step_count)
        print('{} {} {}'.format(episode, step_count, int(np.mean(rewards_100))))

        history.append(step_count)          # 추가된 코드

        # if len(rewards_100) == rewards_100.maxlen:
        #     if np.mean(rewards_100) >= 195:
        #         break

    print(np.mean(rewards_100))

    plt.plot(history)
    plt.show()


# 2013 버전과 동일
def bot_play(model, env):
    state = env.reset()

    reward_sum, done = 0, False
    while not done:
        env.render()
        action = np.argmax(model.predict(np.reshape(state, [1, 4]), verbose=0))
        state, reward, done, _ = env.step(action)
        reward_sum += reward

    print('score :', reward_sum)


env = gym.make('CartPole-v1')

model_main = make_model(env.observation_space.shape[0], env.action_space.n)
model_copy = make_model(env.observation_space.shape[0], env.action_space.n)
copy_weights(model_main, model_copy)

dqn_2015(model_main, model_copy)
bot_play(model_main, env)