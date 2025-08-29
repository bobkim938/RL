import torch
import torch.nn as nn

class network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(network, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # 1st hidden layer
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, input):
        logits = self.linear_relu_stack(input)
        return logits



