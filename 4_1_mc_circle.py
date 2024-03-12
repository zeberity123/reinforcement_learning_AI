# 4_1_mc_circle.py
import numpy as np
import matplotlib.pyplot as plt


# 퀴즈
# 몬테카를로 방식을 사용해서 원의 면적을 구하세요
cnt, n = 0, 1000000
inner, outer = [], []
for i in range(n):
    x = np.random.rand()
    y = np.random.rand()

    v = x ** 2 + y ** 2
    cnt += (v <= 1)

    if v <= 1:
        inner.append((x, y))
    else:
        outer.append((x, y))

print(4 * cnt / n)

# x, y = zip(*[(1, 2), (3, 4), (5, 6)])
# print(len(x), len(y))
# print(x, y)   # (1, 3, 5) (2, 4, 6)

# x, y = zip(*inner)
# print(x[:3])
# print(y[:3])
# print(len(x), len(y))

plt.plot(*list(zip(*inner)), 'ro')
plt.plot(*list(zip(*outer)), 'bo')
plt.show()

