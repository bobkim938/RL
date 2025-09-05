from env import lunarLander
from model import network
from collections import deque
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

def train(numEpisodes):
    global experience_tuple
    env = lunarLander()
    epsilon = 1.0 # initial exploration rate
    discount_rate = 0.99
    reward = []
    episodeCnt = 0
    episodeReward = 0
    replay_buffer = deque(maxlen=10000)

    # Initialize networks and move them to the device
    onlineNet = network(8, 128, 4).to(device)
    targetNet = network(8, 128, 4).to(device)
    targetNet.load_state_dict(onlineNet.state_dict()) # weight transfer to target Net for early stage training stability

    currentState = env.reset()

    while numEpisodes > episodeCnt:
        if len(replay_buffer) < 5000:  # Filling till min size of the replay buffer
            experience_tuple = env.step(torch.randint(0, 4, (1,)).item())
            replay_buffer.append(experience_tuple[0:5])
            currentState = experience_tuple[2]
            if experience_tuple[4] or experience_tuple[5]:
                currentState = env.reset()
        else:
            if len(replay_buffer) == 5000:
                print("Training start")

            if torch.rand(1).item() < epsilon:  # Exploration
                experience_tuple = env.step(torch.randint(0, 4, (1,)).item())  # Random action
                replay_buffer.append(experience_tuple[0:5])
                episodeReward += experience_tuple[3]
            else:  # Exploitation
                logits = onlineNet.forward(torch.tensor(currentState, device=device))
                experience_tuple = env.step(torch.argmax(logits).item())
                replay_buffer.append(experience_tuple[0:5])
                episodeReward += experience_tuple[3]

            currentState = experience_tuple[2]

            # Train network
            mini_batch = random.sample(replay_buffer, 64)
            action_, state_, nextState_, reward_, terminated_ = zip(*mini_batch)  # Unzip mini-batch

            # Convert to tensors and move to the device
            state_ = torch.tensor(np.array(state_), dtype=torch.float32).to(device)
            nextState_ = torch.tensor(np.array(nextState_), dtype=torch.float32).to(device)
            reward_ = torch.tensor(reward_, dtype=torch.float32).to(device)
            action_ = torch.tensor(action_, dtype=torch.int64).unsqueeze(1).to(device)
            terminated_ = torch.tensor(terminated_, dtype=torch.int32).to(device)

            # Compute target and prediction for training
            target = (reward_ + torch.max(targetNet.forward(nextState_), dim=1).values * discount_rate) * (1 - terminated_)
            predict = torch.gather(onlineNet(state_), dim=1, index=action_)

            # Define the loss function and optimizer
            loss_fn = torch.nn.SmoothL1Loss()
            optimizer = torch.optim.Adam(onlineNet.parameters(), lr=0.0001)

            # Compute the loss
            loss = loss_fn(predict.squeeze(1), target)

            # Backpropagation and optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            torch.nn.utils.clip_grad_norm_(onlineNet.parameters(), 1.0) # gradient clipping
            optimizer.step()  # Update weights

            if experience_tuple[4] or experience_tuple[5]:
                if episodeCnt % 10 == 0:
                    targetNet.load_state_dict(onlineNet.state_dict())  # Weight transfer to target Net
                print(f"Episode {episodeCnt} Cumulative Reward: {episodeReward}")
                reward.append(episodeReward)
                episodeReward = 0
                epsilon = max(0.01, epsilon * 0.99)
                episodeCnt += 1
                currentState = env.reset()

    torch.save(onlineNet.state_dict(), "onlineNet.pth")
    fig, ax = plt.subplots()
    ax.plot(reward)
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")

    plt.show()
    fig.savefig("lunarLander-v3_training.png")

train(1000)
