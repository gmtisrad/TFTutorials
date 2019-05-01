import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 10000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9995

class DQNSolver:
    def __init__(self, observation_space, action_space):
        # The lower the exploration rate is, the lower the chance of a random action being taken is.
        self.exploration_rate = EXPLORATION_MAX

        # Action space is the most amount of actions one can take
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Creation and initialization of the ML model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(24, input_shape=(observation_space,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(24, activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_space, activation='linear'))
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # If exploration rate is larger than the random number, take a random action
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # Exploration rate is smaller, so, get the predictions from the model
        q_values = self.model.predict(state)
        # From those predictions, grab the most probable outcome. ie: the best outcome for the model
        return np.argmax(q_values[0])

    # ALLOWS THE NN TO LEARN EVEN WHEN IT ISN'T TOUCHING THE REWARD REGULARLY.
    def experience_replay(self):
        # If actions taken do not exceed the pre-allotted memory set aside, return.
        if len(self.memory) < BATCH_SIZE:
            return

        # Selecting a random batch
        batch = random.sample(self.memory, BATCH_SIZE)

        # This is where the learning happens, based on a random sampling of the episodes.
        for state, action, reward, state_next, terminal in batch:
            # reward is the value to put through your reward function w/ the discount etc...
            q_update = reward
            if not terminal:
                # This creates the update value based on the next state in the case that this isn't the terminal frame/episode
                q_update = reward + GAMMA * np.amax(self.model.predict(state_next)[0])
            # This gets the action state from the model
            q_values = self.model.predict(state)
            # This updates the reward in the state
            q_values[0][action] = q_update
            # This trains the model for the default 1 epoch using the state as the baseline and q_values as the 'True' value.
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def mountain_car():
    # Instantiating the gym
    env = gym.make('MountainCar-v0')
    # Gets the observation space from the environment. (The inputs for the NN)
    observation_space = env.observation_space.shape[0]
    # Gets the action space from the environment. (The outputs for the NN)
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)

    run = 0

    while True:
        run += 1
        # This loop is run with every 'episode' or 'epoch'
        state = env.reset()
        # This sets the shape of the state tensor
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # This loop is run once per frame and is where the NN makes each decision
            env.render()
            # Here, you get the next action from the NN
            action = dqn_solver.act(state)
            # Here,  you get some information returned when you prompt the environment to take the next step. (given by NN)
            state_next, reward, terminal, info = env.step(action)
            # This nullifies the reward if it is in fact the last frame
            reward = reward if not terminal else -reward
            # Reshapes the next state from the env (like before), so that you can put it into the NN
            state_next = np.reshape(state, [1, observation_space])
            # This adds the last frame to the memory of the NN class
            dqn_solver.remember(state, action, reward, state_next, terminal)
            # This steps to the next state and repeats the loop until terminal is returned by the environment.
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: "
                      + str(step) + ', reward: ' + str(reward))
                break
            dqn_solver.experience_replay()

if __name__ == "__main__":
    mountain_car()