import numpy as np
import gym
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make("FrozenLake-v0")
env2 = gym.make('FrozenLakeNotSlippery-v0');

max_epochs = 100000
max_steps = 99

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = .01
epsilon_decay_rate = .0005

learning_rate = .9;

Gamma = .9

qtable = np.zeros((env.observation_space.n, env.action_space.n));

totalReward = 0

for epoch in range(max_epochs):
    state = env.reset()

    for step in range(max_steps):
        random = np.random.uniform(0,1)
        # If random is larger than the epsilon value, predict the best move from the qtable.
        if random > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + Gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        state = new_state

        totalReward += reward

        epsilon = max(min_epsilon, epsilon - (epsilon * epsilon_decay_rate))

        if done is True:
            break
print(qtable)
print('Total Reward: {} \n Epsilon: {}'.format(totalReward, epsilon))

