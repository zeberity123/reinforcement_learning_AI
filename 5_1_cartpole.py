# 5_1_cartpole.py
import gym


env = gym.make('CartPole-v1')

# print(env.observation_space)
# print(env.action_space)

state = env.reset()
print(state)        # [-0.00012604 -0.04882921  0.02284077  0.04876385]

# 퀴즈
# 막대기가 넘어가지 않도록 코드를 조금만 추가해 보세요
done = False
while not done:
    env.render()

    cart_pos, cart_vel, pole_angle, pole_vel = state
    action = 0 if pole_angle < 0 else 1
    state, reward, done, info = env.step(action)
    print(reward, done, info)

env.close()





