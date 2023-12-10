# Script for testing agent performance.
# Can modify the agent used below (e.g to see DQN vs random agent comparison)

import RoadNetEnv
import gymnasium as gym
import numpy as np
import util
import torch
import os
from DQN import DQN
from gymnasium.wrappers.flatten_observation import FlattenObservation

POLICY_NET__DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "policy_net")
# POLICY_NET_FILENAME = "policy_net_multiple_targets.pth"
POLICY_NET_FILENAME = "policy_net_single_target.pth"

NUM_EPISODES = 100
GAMMA = 0.99  

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How the trained DQN agent selects an action. Now greedy
def greedy_select_action(policy_net, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state).max(1).indices.view(1, 1)


# Load environment
graph, locations = util.make_grid_city(5, 5)
env = gym.make('RoadNetEnv-v0', render_mode=None, graph=graph, node_positions=locations, single_target=True)
# Observation space was a dict. Must be flattened for training / testing
env = FlattenObservation(env)

n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Load policy net
policy_net = DQN(n_observations, n_actions)
policy_net.load_state_dict(torch.load(os.path.join(POLICY_NET__DIR_PATH, POLICY_NET_FILENAME)))


# Evaluate performance. No discount factor considered here since we have a small neg. award
# added each timestep
returns = []
observation, info = env.reset()

for _ in range(NUM_EPISODES):
    observation, info = env.reset()
    cur_return = 0
    terminated = truncated = False
    time_step = 0
    # Start of new episode
    while not (terminated or truncated):
        #observation = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Random
        action = env.action_space.sample()  # agent policy that uses the observation and info
        # Policy
        #action = greedy_select_action(policy_net, observation)
        #print(action)

        observation, reward, terminated, truncated, info = env.step(action)

        cur_return += reward

        time_step += 1
        if time_step >= 100:
            break

    # After end of current episode
    print(cur_return)
    returns.append(cur_return)

env.close()
print(returns)
average_return = sum(returns) / len(returns)

print(f"Average return across {NUM_EPISODES} episodes:", average_return)
