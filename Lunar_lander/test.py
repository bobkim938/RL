import torch
from env import lunarLander
from model import network

def run_test(iterations):
    env = lunarLander()
    current_state = torch.tensor(env.return_state())
    model = network(8,128,4)
    model.load_state_dict(torch.load("onlineNet.pth"))
    episodeReward = []
    cumulativeReward = 0
    episodeCnt = 0

    while episodeCnt < iterations:
        action = torch.argmax(model.forward(current_state)).item()
        exp_tuple = env.step(action)
        cumulativeReward += exp_tuple[3]
        current_state = torch.tensor(exp_tuple[2])

        if exp_tuple[4] or exp_tuple[5]:
            episodeReward.append(cumulativeReward)
            cumulativeReward = 0
            current_state = torch.tensor(env.reset())
            episodeCnt+=1

    mean = sum(episodeReward) / len(episodeReward)
    print(f"Mean Reward per Episode for {iterations} episodes: {mean}")

if __name__=='__main__':
    run_test(10)