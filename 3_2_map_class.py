# 3_2_mab_class.py
import numpy as np


# 퀴즈
# 앞에서 만든 코드를 클래스 버전으로 업그레이드 하세요
class Bandit:
    def __init__(self, reward):
        self.mean = 0
        self.sample = 0
        self.reward = reward

    def pull(self):
        return self.reward + np.random.randn()

    def update(self, reward):
        self.sample += 1

        ratio = 1 / self.sample
        self.mean = (1 - ratio) * self.mean + ratio * reward


class Gamer:
    def __init__(self, bandits, is_greedy, e):
        self.is_greedy = is_greedy
        self.means = [b.mean for b in bandits]
        self.e = e
        self.bandits = bandits

    def e_greedy(self):
        if np.random.rand() < self.e:
            return np.random.choice(len(self.means))

        return np.argmax(self.means)

    def greedy(self):
        return np.argmax(self.means)

    def show(self, N):
        for _ in range(N):
            select = self.greedy() if self.is_greedy else self.e_greedy()

            reward = bandits[select].pull()
            bandits[select].update(reward)

        print('value :', [b.reward for b in self.bandits])
        print('mean  :', [b.mean for b in self.bandits])
        print('argmax:', np.argmax([b.mean for b in self.bandits]))


bandits = [Bandit(1.0), Bandit(-2.0), Bandit(3.0)]

g = Gamer(bandits, is_greedy=False, e=0.1)
g.show(N=10000)
