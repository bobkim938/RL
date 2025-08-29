import gymnasium as gym

class lunarLander:
    def __init__(self):
        self.env = gym.make("LunarLander-v3", continuous=False, render_mode='human')
        self.state, self.info = self.env.reset()
        self.reward = None
        self.terminated = None
        self.truncated = None

    def return_state(self):
        return self.state

    def step(self, action):
        prev_state = self.state
        self.state, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        return action, prev_state, self.state, self.reward, self.terminated, self.truncated, self.info

    def reset(self):
        self.state, self.info = self.env.reset()

    

