# 3_6_policy_iteration.py
import numpy as np
np.set_printoptions(precision=2)

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def get_next_state(state, size, action):
    row, col = state

    if action == LEFT and col > 0:
        col -= 1
    elif action == RIGHT and col < size - 1:
        col += 1
    elif action == UP and row > 0:
        row -= 1
    elif action == DOWN and row < size - 1:
        row += 1

    return row, col


def policy_iteration(states, backup, size):
    reward = -1
    grid = np.zeros_like(backup)

    for s in states:
        value = 0
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, size, action)
            value += 0.25 * (reward + backup[next_state])

        grid[s] = value

    return grid


# 퀴즈
# states와 backup을 사용해서 각 state에서의 정책(방향)을 출력하세요
def show_policy(states, backup, size):
    policy = ['    ']
    for s in states:
        rewards = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, size, action)
            rewards.append(backup[next_state])

        max_r = np.max(rewards)

        arrows = ''
        arrows += 'L' if max_r == rewards[LEFT] else ' '
        arrows += 'D' if max_r == rewards[DOWN] else ' '
        arrows += 'R' if max_r == rewards[RIGHT] else ' '
        arrows += 'U' if max_r == rewards[UP] else ' '

        policy.append(arrows)
    policy.append('    ')

    policy = np.reshape(policy, (-1, 4))
    print(policy)


size = 4
backup = np.zeros([size, size])

states = [(i, j) for i in range(size) for j in range(size)]
states.pop(0)
states.pop(-1)

print(states)
print(backup, end='\n\n')

for i in range(155):
    backup = policy_iteration(states, backup, size)
    print(i + 1)
    print(backup, end='\n\n')

show_policy(states, backup, size)



