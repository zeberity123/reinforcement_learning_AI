# 4_7_q_network_tf.py
import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def e_greedy(i, env, vector):
    e = 1 / (i // 100 + 1)
    return env.action_space.sample() if np.random.rand() < e else np.argmax(vector)


def random_noise(i, env, vector):
    return np.argmax(vector + np.random.randn(env.action_space.n) / (i + 1))


def make_onehot(state):
    z = np.zeros(16)
    z[state] = 1
    return z.reshape(1, -1)         # (1, 16)


def q_network_tf():
    env = gym.make('FrozenLake-v1')         # 미끄러운 빙판

    input_size = env.observation_space.n
    output_size = env.action_space.n

    x = tf.placeholder(tf.float32, shape=[1, input_size])
    y = tf.placeholder(tf.float32, shape=[1, output_size])

    w = tf.Variable(tf.random_uniform([input_size, output_size]))
    b = tf.Variable(tf.zeros([4]))

    hx = tf.matmul(x, w) + b            # Dense layer

    loss = tf.reduce_mean((hx - y) ** 2)    # mse
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)   # SGD

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    success, discounted = 0, 0.99
    for i in range(2000):
        state = env.reset()
        # env.render()

        done = False
        while not done:
            # p = model.predict(make_onehot(state), verbose=0)    # (1, 4)
            p = sess.run(hx, {x: make_onehot(state)})
            action = random_noise(i, env, p[0])
            next_state, reward, done, _ = env.step(action)

            if done:
                p[0, action] = reward
            else:
                # p_next = model.predict(make_onehot(next_state), verbose=0)
                p_next = sess.run(hx, {x: make_onehot(next_state)})
                p[0, action] = reward + discounted * np.max(p_next)

            # model.fit(make_onehot(state), p, epochs=1, verbose=0)
            sess.run(train, {x: make_onehot(state), y: p})
            state = next_state

            # env.render()
        success += reward

        if i % 10 == 0:
            print(i, '성공 :', success, success / (i+1))



    sess.close()


q_network_tf()








