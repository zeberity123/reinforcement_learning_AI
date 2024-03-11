# 3_5_frozen_lake_game.py
import readchar
import gym


# 퀴즈
# readchar 모듈을 사용해서
#  asdf 키에 따라 움직이는 코드를 만드세요
# (LDRU)
# id_name = "MyLake-v1"
# gym.envs.register(id_name, entry_point='gym.envs.toy_text:FrozenLakeEnv')
#
# env = gym.make(id_name, map_name='4x4', is_slippery=False)
env = gym.make('FrozenLake-v0')

env.reset()
env.render()

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
# arrows = {'a': LEFT, 's': DOWN, 'd': RIGHT, 'f': UP}
arrows = {
    # '\x1b[D': LEFT
    # '\x1b[B': DOWN,
    # '\x1b[C': RIGHT,
    # '\x1b[A': UP,
    '\x00K': LEFT,
    '\x00P': DOWN,
    '\x00M': RIGHT,
    '\x00H': UP,
}

done = False
while not done:
    c = readchar.readkey()
    action = arrows[c]

    _, _, done, _ = env.step(action)
    env.render()

env.close()

