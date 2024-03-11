# 3_3_mab_final.py
import numpy as np
import matplotlib.pyplot as plt


# 퀴즈
# 숫자 리스트에 대해 누적 합계를 구하세요
#    1  3 10 14 19
def show_cumulated():
    a = [1, 2, 7, 4, 5]

    cum = []
    # for i in range(len(a)):
    #     s = 0
    #     for j in range(i+1):
    #         s += a[j]
    #
    #     cum.append(s)

    # for i in range(len(a)):
    #     cum.append(sum(a[:i+1]))

    s = 0
    for i in range(len(a)):
        s += a[i]
        cum.append(s)

    print(cum)
    print(np.cumsum(a))


class Bandit:
    def __init__(self, reward):
        self.mean = 10
        self.sample = 0
        self.reward = reward

    def pull(self):
        return self.reward + np.random.randn()

    def update(self, reward):
        self.sample += 1

        ratio = 1.0 / self.sample
        self.mean = (1 - ratio) * self.mean + ratio * reward


class Gamer:
    def __init__(self, bandits, is_greedy, e):
        self.is_greedy = is_greedy
        self.means = [b.mean for b in bandits]
        self.e = e
        self.bandits = bandits

    def e_greedy(self):
        # if np.random.rand() < self.e:
        #     return np.random.choice(len(self.means))

        return np.argmax(self.means)

    def greedy(self):
        return np.argmax(self.means)

    def show(self, N):
        rewards = []
        for _ in range(N):
            select = self.greedy() if self.is_greedy else self.e_greedy()

            reward = self.bandits[select].pull()
            self.bandits[select].update(reward)

            rewards.append(reward)

        print('value :', [b.reward for b in self.bandits])
        print('mean  :', [b.mean for b in self.bandits])
        print('argmax:', np.argmax([b.mean for b in self.bandits]))

        # 퀴즈
        # 모든 슬롯머신에서 나온 결과를 누적합계를 사용한 그래프를 그리세요
        avg = np.cumsum(rewards) / (np.arange(N) + 1)
        plt.xscale('log')
        plt.plot(avg, 'r')
        plt.show()


def mab_final():
    bandits = [Bandit(1.0), Bandit(2.0), Bandit(3.0)]

    g = Gamer(bandits, is_greedy=True, e=0.1)
    g.show(N=10000)


# show_cumulated()
mab_final()

