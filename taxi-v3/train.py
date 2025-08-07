from agent import agent
from env import taxi_v3
import matplotlib.pyplot as plt

def train(epoch):
    env = taxi_v3()
    driver = agent(500, 6) # for taxi-v3

    episodeReward = []
    epsilon = [0.2]

    episode = 0
    total_reward = 0

    fig, ax = plt.subplots(1, 2)

    while episode < epoch:
        state = env.return_state()
        action = driver.choose_action(state)
        nextState, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        driver.update_qtable(reward, action, state, nextState)

        if terminated or truncated:
            episodeReward.append(total_reward)
            epsilon.append(driver.epsilon_decay())
            total_reward = 0
            env.reset()
            episode+=1

    ax[0].plot(episodeReward)
    ax[0].set_title("Episode Reward")
    ax[0].set_xlabel("Episodes")
    ax[0].set_ylabel("Reward")

    ax[1].plot(epsilon)
    ax[1].set_title("Epsilon")
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylabel("Epsilon")

    plt.show()
    fig.savefig("taxi-v3_training.png")
    driver.save_qTable()

if __name__ == "__main__":
    train(1000)
