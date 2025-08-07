import gymnasium as gym

'''
Env: Taxi-v3
Action Space: Discrete(6)
Observation Space: Discrete(500), ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
rewards: -1 per step unless other reward is triggered
         20 delivering passenger
         -10 executing "pickup" and "drop-off" actions illegally
'''

class taxi_v3:
    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode='ansi')
        self.state, self.info = self.env.reset()
        self.reward = None
        self.terminated = None
        self.truncated = None

    def return_state(self):
        return self.state

    def step(self, action):
        self.state, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        # print(self.env.render())
        return self.state, self.reward, self.terminated, self.truncated, self.info

    def reset(self):
        self.state, self.info = self.env.reset()
        # print(self.env.render())

