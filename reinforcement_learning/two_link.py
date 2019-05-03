import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import random
from gym.envs.registration import register
GAMMA = .95
LEARNING_RATE = .95

MEMORY_MAX = 10000000
BATCH_MAX = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = .01
EXPLORATION_DECAY = .9995


class QDN_solve:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = EXPLORATION_MAX

        self.memory = deque(maxlen=MEMORY_MAX)
        self.wins = 0

        self.model = keras.Sequential()

        self.model.add(keras.layers.Dense(256, input_shape=(observation_space.shape[0],), activation='relu'))
        self.model.add(keras.layers.Dense(512, activation='sigmoid'))
        self.model.add(keras.layers.Dense(self.action_space, activation='relu'))

        # Compiles the NN with the mean squared error loss function and the Adam reinforcement optimizer algorithm
        self.model.compile(loss=tf.losses.mean_squared_error,
                           optimizer=tf.keras.optimizers.RMSprop(lr=LEARNING_RATE))

    def remember(self, state, reward, next_state, action, done):
        self.memory.append((state, reward, next_state, action, done))

    def act(self, state):
        # If exploration rate is larger than the random number, act at random. (Exploring the action space)
        state = np.reshape(state, (1,self.observation_space.shape[0]))
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # Get the q_values for the current state from the NN. (ie: The recommended moves based on P() of being a good move)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    # Experience replay is where the reinforcement algorithm comes in.
    def experience_replay(self):
        if len(self.memory) < BATCH_MAX:
            return

        batch = random.sample(self.memory, BATCH_MAX)# TRUE RANDOM SAMPLING

        # Creates updated q_values for each value in the batch.
        for state, reward, next_state, action, terminal in batch:
            q_update = reward

            next_state = np.reshape(next_state, (1,self.observation_space.shape[0]))
            state = np.reshape(state, (1,self.observation_space.shape[0]))

            q_values = self.model.predict(state)

            if not terminal:
                # QVal update algorithm
                # qtable[state, action] = qtable[state, action] + learning_rate * (reward + Gamma * np.max(qtable[new_state, :]) - qtable[state, action])
                q_update = q_values[0][action] + LEARNING_RATE *(reward + GAMMA * np.amax(self.model.predict(next_state)[0]) - q_values[0][action])



            q_values[0][action] = q_update

            self.model.fit(state, q_values, verbose=0)

            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def acrobot():
    env = gym.make('Acrobot-v1')
    observation_space = env.observation_space
    action_space = env.action_space.n

    dqn_solver = QDN_solve(action_space, observation_space)

    episode = 0
    while True:
        episode += 1

        state = env.reset()

        rewardTot = 0
        step = 0
        while True:
            step += 1
            env.render()

            action = dqn_solver.act(state)

            next_state, reward, done, info = env.step(action)

            rewardTot += reward

            reward = reward if not done else -reward

            dqn_solver.remember(state, reward, next_state, action, done)

            state = next_state

            if 0 is step % 10:
                print("Episode: " + str(episode) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " +
                      str(step) + ", reward: " + str(reward) + ", rewardTot: " + str(rewardTot))
                dqn_solver.experience_replay()
                if done:
                    print('\n\nDONE\n\n')
                    break



if __name__ == '__main__':
    acrobot()