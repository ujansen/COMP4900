import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import torch.nn.functional as F
from main import make_grid_city
import RoadNetEnv

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

graph, locations = make_grid_city(5, 5)
env = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations)

state_dim = 4
action_dim = 5

loaded_policy_net = PolicyNetwork(state_dim, action_dim)
loaded_policy_net.load_state_dict(torch.load("ppo_policy_batch.pt"))

loaded_value_net = ValueNetwork(state_dim)
loaded_value_net.load_state_dict(torch.load("ppo_value_batch.pt"))

state, _ = env.reset()
print(state)
total_reward = 0

while True:
    with torch.no_grad():
        flat_state = np.concatenate([np.atleast_1d(state['agent']), np.atleast_1d(state['target']),
                                     np.atleast_1d(state['prev']),
                                     np.atleast_1d(state['traffic'])])
        action_probs = loaded_policy_net(torch.FloatTensor(flat_state))
        action = np.argmax(action_probs.numpy())

    next_state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    state = next_state

    if terminated:
        break

print(f"Total reward during testing: {total_reward}")