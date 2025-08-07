# Taxi-V3 using Q-Learning

## Overview:
- RL agent to solve Taxi-v3 environment using Q-learning algorithm.
- Average Reward per episode is around 8.0 after training.

## Environment:
https://gymnasium.farama.org/environments/toy_text/taxi/

## Dependencies:
- gymnasium
- numpy
- matplotlib

## How to Run:
1. Install the required dependencies:
   ```bash
   pip install gymnasium numpy matplotlib
   ```
2. Clone the repository:
   ```bash
   https://github.com/bobkim938/RL.git
   ```
3. Navigate to the `taxi-v3` directory:
   ```bash
   cd RL/taxi-v3/
   ```
4. To train the agent, delete saved Q-table and run train.py:
   ```bash
    rm qTable.npy
    python train.py
    ```
5. Training Result saved as:
   ```bash
   taxi-v3_training.png
    ```
6. To test the agent, run test.py:
    ```bash
   python test.py
   ```
