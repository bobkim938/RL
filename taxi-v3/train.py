from agent import agent
from env import taxi_v3

def train(epoch):
    env = taxi_v3()
    driver = agent(500, 6) # for taxi-v3

    cnt = 0
    while cnt < epoch:
        state = env.return_state()
        action = driver.choose_action(state)
        nextState, reward, terminated, truncated, info = env.step(action)
        driver.update_qtable(reward, action, state, nextState)
        if terminated or truncated:
            print("new episode")
            driver.epsilon_decay()
            env.reset()
            cnt+=1


if __name__ == "__main__":
    train(1000)
