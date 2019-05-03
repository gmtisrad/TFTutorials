import numpy as np
import tensorflow as tf
import importance_sampling.training
import random
import gym
import heapq

MAX_MEMORY = 20000
MAX_BATCH = 5000

MIN_WEIGHT_EXP = 0.01
MAX_WEIGHT_EXP = 1.0
WEIGHT_EXP = 0.01

LEARNING_RATE = 1.0
MAX_LEARNING_RATE = 1.0
MIN_LEARNING_RATE = .01
LEARNING_DECAY_RATE = .9995

GAMMA = .8

MAX_EPISODES = 500



class DQNSolver:
    def __init__(self, action_space, observation_space):
        self.model = tf.keras.sequential()
        self.model.add(tf.keras.layers.Dense(64, input_shape=(observation_space, 1), activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
        self.model.add(tf.keras.layers.Dense(action_space.n), activation='relu')
        self.model.compile(loss='mse', optimization='RMSProp')
        self.priorityQueue = []

    def remember(self, state, action, reward, last_state, last_action, priority):
        heapq.heappush(self.priorityQueue, (priority, state, action, reward, last_state, last_action))

    def computeTDError(self, state, action, reward, last_state, last_action):
        temporal_difference_error = reward + GAMMA * self.model.predict(state)[np.argmax(self.model.predict(state))] - self.model.predict(last_state)[last_action]
        return temporal_difference_error

    def calculateISWeight(self, state, action):
        importance_sampling_weight = np.sqrt(np.exp((MAX_MEMORY * self.model.predict(state)[action]), (-1 * WEIGHT_EXP))/MAX_EPISODES)
        return importance_sampling_weight

    def act(self, state):
        # If exploration rate is larger than the random number, act at random. (Exploring the action space)
        state = np.reshape(state, (1,self.observation_space.shape[0]))
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # Get the q_values for the current state from the NN. (ie: The recommended moves based on P() of being a good move)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])


def main():
    env = gym.make('Acrobot-v1')
    dqn = DQNSolver(env.action_space, env.observation_space)
    last_state = None
    last_action = None

    for episode in range(MAX_EPISODES):
        state = env.reset()

        for step in range(500):
            action = dqn.act(state)

            next_state, reward, done, _ = env.step(action)