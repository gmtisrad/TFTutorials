import numpy as np
import tensorflow as tf
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
        self.observation_space = observation_space
        self.action_space = action_space
        self.exploration_rate = 1.0
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(64, input_shape=(observation_space.shape[0],), activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
        self.model.add(tf.keras.layers.Dense(action_space.n, activation='relu'))
        self.model.compile(loss='mse', optimizer='RMSProp')
        self.priorityQueue = []

    def remember(self, priority, state, action, reward, last_state, last_action):
        heapq.heappush(self.priorityQueue, (priority, state, action, reward, last_state, last_action))

    def computeTDError(self, state, action, reward, last_state, last_action):
        state = np.reshape(state, (self.observation_space.shape[0],1))
        last_state = np.reshape(last_state, (self.observation_space.shape[0],1))

        temporal_difference_error = reward + GAMMA * self.model.predict(state)[np.argmax(self.model.predict(state))] - self.model.predict(last_state)[last_action]
        return temporal_difference_error

    def calculateISWeight(self, state, action):
        importance_sampling_weight = np.sqrt(np.exp((MAX_MEMORY * self.model.predict(state)[action]), (-1 * WEIGHT_EXP))/MAX_EPISODES)
        return importance_sampling_weight

    def act(self, state):
        # If exploration rate is larger than the random number, act at random. (Exploring the action space)
        state = np.reshape(state, (self.observation_space.shape[0],))
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space.n)
        # Get the q_values for the current state from the NN. (ie: The recommended moves based on P() of being a good move)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.priorityQueue) < MAX_MEMORY:
            return

        batch = heapq.nlargest(self.priorityQueue, MAX_BATCH)

        # Creates updated q_values for each value in the batch.
        for priority, state, action, reward, last_state, last_action in batch:
            q_update = reward

            next_state = np.reshape(next_state.n, (self.observation_space.shape[0],1))
            state = np.reshape(state, (self.observation_space.shape[0],1))

            q_values = self.model.predict(state)

            # QVal update algorithm
            # qtable[state, action] = qtable[state, action] + learning_rate * (reward + Gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            q_update = q_values[0][action] + LEARNING_RATE *  (reward + GAMMA * np.amax(self.model.predict(next_state)[0]) - q_values[0][action])


            q_values[0][action] = q_update

            self.model.fit(state, q_values, verbose=0)

            self.exploration_rate *= LEARNING_DECAY_RATE
            self.exploration_rate = max(MIN_LEARNING_RATE, self.exploration_rate)

def main():
    env = gym.make('Acrobot-v1')
    dqn = DQNSolver(env.action_space, env.observation_space)
    last_state = None
    last_action = None

    for episode in range(MAX_EPISODES):
        state = env.reset()

        for step in range(500):
            env.render()

            action = dqn.act(state)

            next_state, reward, done, _ = env.step(action)

            if last_state is not None and last_action is not None:
                priority = dqn.computeTDError(state, action, reward, last_state, last_action)
                dqn.remember((priority, state, action, reward, last_state, last_action,))

            last_state = state
            last_action = action
            state = next_state

if __name__ == "__main__":
    main()