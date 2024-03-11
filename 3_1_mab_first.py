# 3_1_mab_first.py
import numpy as np


def e_greedy(means, e):
    # if np.random.rand() < e:
    #     return np.random.choice(len(means))

    return np.argmax(means)


def show_e_greedy(bandits, epsilon, N):
    means = [0] * len(bandits)
    samples = [0] * len(bandits)

    for _ in range(N):
        select = e_greedy(means, epsilon)
        # print(select)

        samples[select] += 1
        ratio = 1 / samples[select]

        reward = bandits[select] + np.random.randn()
        means[select] = (1 - ratio) * means[select] + ratio * reward

    print('value :', bandits)
    print('mean  :', means)
    print('argmax:', np.argmax(means))


bandits = [1.0, -2.0, 3.0]
show_e_greedy(bandits, epsilon=0.1, N=10000)

# for i in range(100):
#     print(np.random.rand())       # 균등분포
#     print(np.random.choice(5))
#     print(np.random.randn())      # 정규분포



