# 3_4_frozen_lake.py
import gym          # 0.17.3


def gym_basic():
    env = gym.make('FrozenLake-v0')

    env.reset()
    env.render()

    result = env.step(1)
    env.render()

    print(result)
    #  state reward Done(terminated) info
    # (0,    0.0,   False,          {'prob': 0.3333333333333333})
    # 0.26 이후 버전에는 truncated 반환값 추가


def gym_custom_env():
    id_name = "MyLake-v1"
    gym.envs.register(id_name, entry_point='gym.envs.toy_text:FrozenLakeEnv')

    env = gym.make(id_name, map_name='4x4', is_slippery=False)

    env.reset()
    env.render()

    # 퀴즈
    # step 함수를 호출해서 Goal까지 이동해 보세요
    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

    for action in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
        _, _, done, _ = env.step(action)
        env.render()

        print('   done :', done)


# gym_basic()
gym_custom_env()

