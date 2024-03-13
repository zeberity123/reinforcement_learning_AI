# 5_3_cartpole_q_net_tf.py
import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# 퀴즈
# 케라스 버전을 텐서플로 버전으로 수정하세요
def e_greedy(i, env, vector):
    e = 1 / (i // 100 + 1)
    return env.action_space.sample() if np.random.rand() < e else np.argmax(vector)


def random_noise(i, env, vector):
    return np.argmax(vector + np.random.randn(env.action_space.n) / (i + 1))


def q_network_tf():
    env = gym.make('CartPole-v1')

    input_size = 4
    output_size = 2

    x = tf.placeholder(tf.float32, shape=[1, input_size])
    y = tf.placeholder(tf.float32, shape=[1, output_size])

    w = tf.Variable(tf.random_uniform([input_size, output_size]))
    b = tf.Variable(tf.zeros([2]))

    hx = tf.matmul(x, w) + b            # Dense layer

    loss = tf.reduce_mean((hx - y) ** 2)    # mse
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)   # SGD

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    result, discounted = [], 0.9
    for i in range(2000):
        state = env.reset()
        # env.render()

        done, step_count = False, 0
        while not done:
            p = sess.run(hx, {x: np.reshape(state, [1, -1])})
            action = random_noise(i, env, p[0])
            next_state, reward, done, _ = env.step(action)
            step_count += 1

            if done:
                p[0, action] = -100
            else:
                p_next = sess.run(hx, {x: np.reshape(next_state, [1, -1])})
                p[0, action] = reward + discounted * np.max(p_next)

            sess.run(train, {x: np.reshape(state, [1, -1]), y: p})
            state = next_state

            env.render()

        result.append(step_count)

        if i % 10 == 9:
            print(i+1, '횟수 :', np.mean(result[-10:]))

    sess.close()


q_network_tf()
