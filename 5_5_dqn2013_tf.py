# 5_5_dqn2013_tf.py
import numpy as np
import random
import gym
from collections import deque

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def make_model(input_size, output_size):
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    y = tf.placeholder(tf.float32, shape=[None, output_size])

    w1 = tf.Variable(tf.random_uniform([input_size, 16]))
    b1 = tf.Variable(tf.zeros([16]))

    w2 = tf.Variable(tf.random_uniform([16, output_size]))
    b2 = tf.Variable(tf.zeros([output_size]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)
    hx = tf.matmul(r1, w2) + b2

    loss = tf.reduce_mean((hx - y) ** 2)    # mse
    train = tf.train.AdamOptimizer(0.01).minimize(loss)   # SGD

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    return x, y, hx, train, sess


def annealing_epsilon(episode, min_e, max_e, target_episode):
    # episode가 50을 넘으면 0 반환. min_e : 0, max_e : 1
    slope = (min_e - max_e) / target_episode        # -0.02 = -1 / 50
    return max(min_e, slope * episode + max_e)      # max(0, -0.02 * epi + 1)


def dqn_2013(x, y, hx, train, sess):
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
                # action = np.argmax(model.predict(np.reshape(state, [1, -1]), verbose=0))
                action = np.argmax(sess.run(hx, {x: np.reshape(state, [-1, 4])}))

            next_state, reward, done, _ = env.step(action)

            if done:
                reward = -1

            replay_buffer.append((state, next_state, action, reward, done))

            state = next_state
            step_count += 1

            if len(replay_buffer) > 64:
                minibatch = random.sample(replay_buffer, 64)    # 2 28 20

                states = np.vstack([x[0] for x in minibatch])
                next_states = np.vstack([x[1] for x in minibatch])
                actions = np.array([x[2] for x in minibatch])
                rewards = np.array([x[3] for x in minibatch])
                dones = np.array([x[4] for x in minibatch])

                xx = states
                # yy = model.predict(xx, verbose=0)
                yy = sess.run(hx, {x: xx})

                # next_rewards = model.predict(next_states, verbose=0)
                next_rewards = sess.run(hx, {x: np.reshape(next_states, [-1, 4])})

                yy[range(len(xx)), actions] = rewards + 0.99 * np.max(next_rewards, axis=1) * ~dones

                # model.fit(xx, yy, batch_size=16, verbose=0)
                n = 0
                sess.run(train, {x: xx[n:n+16], y: yy[n:n+16]})
                n += 16
                sess.run(train, {x: xx[n:n+16], y: yy[n:n+16]})
                n += 16
                sess.run(train, {x: xx[n:n+16], y: yy[n:n+16]})
                n += 16
                sess.run(train, {x: xx[n:n+16], y: yy[n:n+16]})

        rewards_100.append(step_count)
        print('{} {} {}'.format(episode, step_count, int(np.mean(rewards_100))))

        if len(rewards_100) == rewards_100.maxlen:
            if np.mean(rewards_100) >= 195:
                break

    print(np.mean(rewards_100))


def bot_play(x, hx, sess, env):
    state = env.reset()

    reward_sum, done = 0, False
    while not done:
        env.render()
        # action = np.argmax(model.predict(np.reshape(state, [1, 4]), verbose=0))
        p = sess.run(hx, {x: np.reshape(state, [1, 4])})
        action = np.argmax(p)
        state, reward, done, _ = env.step(action)
        reward_sum += reward

    print('score :', reward_sum)


env = gym.make('CartPole-v1')
# env = gym.wrappers.Monitor(env, directory="gym-results/")     # ffmpeg으로 녹화. 추가 설치가 필요하다

x, y, hx, train, sess = make_model(env.observation_space.shape[0], env.action_space.n)
dqn_2013(x, y, hx, train, sess)
bot_play(x, hx, sess, env)
