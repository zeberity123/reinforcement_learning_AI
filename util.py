# util.py
import numpy as np


def insert_text(lines, row, col, text):
    line_row = row * 3 + 1
    line_col = col * 12 + 4

    s = lines[line_row]
    lines[line_row] = s[:line_col] + text + s[line_col+len(text):]


# 16 x 4
def draw_q_table(q_table, holes=None):
    q1 = np.zeros([len(q_table), 3, 3])
    for v, t in zip(q1, q_table):
        v[1, 0] = t[0]              # LEFT
        v[2, 1] = t[1]              # DOWN
        v[1, 2] = t[2]              # RIGHT
        v[0, 1] = t[3]              # UP

    q2 = q1.reshape(4, 4, 3, 3)     # 0, 4, 3, 3

    lines = []
    for i in range(4):
        for j in range(3):
            flat = np.reshape(q2[i, :, j, :], newshape=-1)

            concat = ['{:.2f}'.format(r) if r else ' ** ' for r in flat]
            lines.append(''.join(concat))

    if holes:
        for r, c in holes:
            insert_text(lines, r, c, 'HOLE')

    insert_text(lines, 0, 0, 'STRT')
    insert_text(lines, 3, 3, 'GOAL')

    print('+' + '-' * (12 * 4 + 3 * (4 - 1) + 1) + '+')
    for i, line in enumerate(lines):
        print('+', end='')
        for j, c in enumerate(line):
            print(c, end='')
            if j % 12 == 11:
                print(' | ', end='')
        print()
        if i % 3 == 2:
            print('+' + '-' * (12 * 4 + 3 * (4 - 1) + 1) + '+')


