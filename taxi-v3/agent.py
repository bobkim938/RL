import numpy as np
import os

class agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_table = np.zeros((observation_space, action_space))
        self.alpha = 0.7 # learning rate
        self.gamma = 0.95 # discount factor
        self.epsilon = 0.2 # exploration rate
        self.epsilon_min = 0.01

        # Load q_table if the qTable exists
        if os.path.exists('qTable.npy'):
            self.q_table = np.load('qTable.npy')
            self.epsilon = 0 # for inference mode
            print("Loaded q_table from qTable.npy")

    def choose_action(self, state):
        if np.random.rand() < self.epsilon: # perform random action
            return np.random.randint(0, self.action_space)
        else: # choose action that maximizes state-action value
            return np.argmax(self.q_table[state])

    def update_qtable(self, reward, action, state, next_state):
        # Q(S,A) <- Q(S,A) + a(R + y*Q(S',A') - Q(S,A))
        self.q_table[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])

    def epsilon_decay(self):
        # step-based exponential decay
        self.epsilon = max(self.epsilon_min, self.epsilon * 0.98)
        return self.epsilon

    def save_qTable(self):
        np.save('qTable.npy', self.q_table)

