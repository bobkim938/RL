from agent import agent
from env import taxi_v3
import numpy as np

def run_test(numEpisodes):
    env = taxi_v3()
    driver = agent(500, 6)

    episodesCnt = 0
    episodeReward = []

    total_reward = 0

    while episodesCnt < numEpisodes:
        state = env.return_state()
        action = driver.choose_action(state)
        nextState, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            episodeReward.append(total_reward)
            print(f"Total Reward for Episode {episodesCnt}: {total_reward}")
            env.reset()
            total_reward = 0
            episodesCnt+=1

    print(f"Average Reward for 10 Episodes: {np.mean(episodeReward)}")


if __name__ == "__main__":
    run_test(10)