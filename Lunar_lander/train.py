from env import lunarLander
from model import network
from collections import deque
import torch
import random

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

def train(numEpisodes):
    global experience_tuple
    env = lunarLander()
    epsilon = [0.2]
    reward = []
    episodeCnt = 0
    episodeReward = 0
    replay_buffer = deque(maxlen=100000)

    onlineNet = network(8, 512, 4)
    targetNet = network(8, 512, 4)

    while numEpisodes > episodeCnt:
        if len(replay_buffer) < 100: # filling till min size of the replay buffer
            experience_tuple = env.step(torch.randint(0, 4, (1,)).item())
            replay_buffer.append(experience_tuple[0:3])
            if experience_tuple[3] or experience_tuple[4]:
                env.reset()
        else:
            if torch.rand(1).item() < epsilon[episodeCnt]: # exploration
                experience_tuple = env.step(torch.randint(0, 4, (1,)).item()) # random action
                replay_buffer.append(experience_tuple[0:3])
            else: # exploitation
                logits = onlineNet.forward(torch.tensor(experience_tuple[1]))
                experience_tuple = env.step(torch.argmax(logits).item())
                replay_buffer.append(experience_tuple[0:3])

            # train network
            mini_batch = random.sample(replay_buffer, 64)
            state, nextState, reward = zip(*mini_batch) # unzip mini-batch
            




            if experience_tuple[3] or experience_tuple[4]:
                reward.append(episodeReward)
                episodeCnt+=1
                env.reset()


train(1000)
