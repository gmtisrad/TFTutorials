import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import random
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100000,
    reward_threshold=0.8196, # optimum = .8196
)

GAMMA = .95
LEARNING_RATE = 1.0

MEMORY_MAX = 10000000
BATCH_MAX = 10

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = .01
EXPLORATION_DECAY = .995


class QDN_solve:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = EXPLORATION_MAX

        self.memory = deque(maxlen=MEMORY_MAX)
        self.wins = 0

        self.model = keras.Sequential()

        self.model.add(keras.layers.Dense(128, input_dim=1, activation='relu'))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(self.action_space.n, activation='linear'))

        # Compiles the NN with the mean squared error loss function and the Adam reinforcement optimizer algorithm
        self.model.compile(loss=tf.losses.mean_squared_error,
                           optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE))

    def remember(self, state, reward, next_state, action, done):
        self.memory.append((state, reward, next_state, action, done))

    def act(self, state):
        # If exploration rate is larger than the random number, act at random. (Exploring the action space)
        state = np.reshape(state, (1,))
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space.n)
        # Get the q_values for the current state from the NN. (ie: The recommended moves based on P() of being a good move)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    # Experience replay is where the reinforcement algorithm comes in.
    def experience_replay(self):
        if len(self.memory) < BATCH_MAX:
            return

        batch = random.sample(self.memory, BATCH_MAX)

        # Creates updated q_values for each value in the batch.
        for state, reward, next_state, action, terminal in batch:
            q_update = reward

            next_state = np.reshape(next_state, (1,))
            state = np.reshape(state, (1,))

            if not terminal:
                q_update = reward + GAMMA * np.amax(self.model.predict(next_state)[0])

            q_values = self.model.predict(state)

            q_values[0][action] = q_update

            self.model.fit(state, q_values, verbose=0)

            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def frozen_lake_4x4():
    env = gym.make('FrozenLake-v0')
    observation_space = env.observation_space
    action_space = env.action_space

    dqn_solver = QDN_solve(action_space, observation_space)

    episode = 0
    while True:
        episode += 1

        state = env.reset()

        step = 0
        while True:
            step += 1
            # env.render()

            action = dqn_solver.act(state)

            next_state, reward, done, info = env.step(action)

            reward = reward if not done else -reward

            dqn_solver.remember(state, reward, next_state, action, done)

            state = next_state

            if done:
                if reward != 0:
                    env.render()
                    dqn_solver.wins += 1
                print("Episode: " + str(episode) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " +
                      str(step) + ", wins: " + str(dqn_solver.wins) + ', ratio: ' + str(dqn_solver.wins/episode))
                break

            dqn_solver.experience_replay()




if __name__ == '__main__':
    frozen_lake_4x4()