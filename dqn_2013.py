import numpy as np
import random
import gym
from collections import deque
import keras


def make_model(input_size, output_size):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([input_size]))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(output_size))

    model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.mse)
    return model


def annealing_epsilon(episode, min_e, max_e, target_episode):
    # episode가 50을 넘으면 0 반환. min_e : 0, max_e : 1
    slope = (min_e - max_e) / target_episode        # -0.02 = -1 / 50
    return max(min_e, slope * episode + max_e)      # max(0, -0.02 * epi + 1)


def dqn_2013(model):
    replay_buffer = deque(maxlen=50000)
    rewards_100 = deque(maxlen=100)

    max_episodes = 500
    for episode in range(max_episodes):
        state = env.reset()

        e = annealing_epsilon(episode, 0.0, 1.0, max_episodes / 10)
        done, step_count, loss = False, 0, 0
        while not done:
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(np.reshape(state, [1, -1]), verbose=0))

            next_state, reward, done, _ = env.step(action)

            if done:
                reward = -1

            replay_buffer.append((state, next_state, action, reward, done))

            state = next_state
            step_count += 1

            if len(replay_buffer) > 64:
                minibatch = random.sample(replay_buffer, 64)    # 2 28 20
                # print(len(minibatch))
                # print(len(minibatch[0]))
                # print(len(minibatch[0][0]))
                # print(len(minibatch[0][1]))
                # print(minibatch[0][2])
                # print(minibatch[0][3])
                # print(minibatch[0][4])

                states = np.vstack([x[0] for x in minibatch])
                next_states = np.vstack([x[1] for x in minibatch])
                actions = np.array([x[2] for x in minibatch])
                rewards = np.array([x[3] for x in minibatch])
                dones = np.array([x[4] for x in minibatch])

                xx = states
                yy = model.predict(xx, verbose=0)

                next_rewards = model.predict(next_states, verbose=0)

                yy[range(len(xx)), actions] = rewards + 0.99 * np.max(next_rewards, axis=1) * ~dones

                model.fit(xx, yy, batch_size=16, verbose=0)

        rewards_100.append(step_count)
        print('{} {} {}'.format(episode, step_count, int(np.mean(rewards_100))))

        if len(rewards_100) == rewards_100.maxlen:
            if np.mean(rewards_100) >= 195:
                break

    print(np.mean(rewards_100))


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
# env = gym.wrappers.Monitor(env, directory="gym-results/")     # ffmpeg으로 녹화. 추가 설치가 필요하다

model = make_model(env.observation_space.shape[0], env.action_space.n)
dqn_2013(model)
bot_play(model, env)
