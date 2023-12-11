import RoadNetEnv
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


def make_grid_city(dim_x, dim_y):
    """
    Creates a grid-like environment of size (dim_x x dim_y)

    :param dim_x: Rows in the grid environment
    :param dim_y: Columns in the grid environment
    :return: graph: A numpy array of size (dim_x x dim_y, 5) where every row contains a node,
                    and its edge list. A node's edge list is basically the nodes it can move to.
             locations: A numpy array of size (dim_x x dim_y, 2) where every row contains a node,
                    and its position in 2D as (x, y) coordinates.
    """
    graph = np.zeros((dim_x * dim_y, 5), dtype=int)
    locations = np.zeros((dim_x * dim_y, 2), dtype=float)
    offset_per_x = 1.0 / dim_x
    offset_per_y = 1.0 / dim_y
    for x in range(dim_x):
        for y in range(dim_y):
            index = y * dim_x + x
            locations[index][0] = x * offset_per_x + (offset_per_x / 2)
            locations[index][1] = y * offset_per_y + (offset_per_y / 2)

            if x > 0:
                graph[index][0] = y * dim_x + (x - 1)
            else:
                graph[index][0] = index

            if y > 0:
                graph[index][1] = (y - 1) * dim_x + x
            else:
                graph[index][1] = index

            if x < dim_x - 1:
                graph[index][2] = y * dim_x + (x + 1)
            else:
                graph[index][2] = index

            if y < dim_y - 1:
                graph[index][3] = (y + 1) * dim_x + x
            else:
                graph[index][3] = index

            graph[index][4] = index

    return graph, locations


class Policy(nn.Module):
    def __init__(self, obs_space, output_size):
        super(Policy, self).__init__()
        input_size = 0
        self.flat_layers = nn.ModuleList()
        self.feature_keys = []
        for space_name, space in obs_space.spaces.items():
            self.feature_keys.append(space_name)
            if isinstance(space, gym.spaces.Discrete):
                input_size += space.n
                self.flat_layers.append(nn.Embedding(space.n, space.n))  # Using 10 as an example embedding size
            elif isinstance(space, gym.spaces.MultiBinary):
                input_size += space.shape[0]
                self.flat_layers.append(nn.Linear(space.shape[0], space.shape[0]))  # Example linear layer

        self.fc = nn.Linear(input_size, 128)
        self.fc_combined = nn.Linear(128, 128)
        # self.fc_combined_2 = nn.Linear(128, 256)
        self.fc_output = nn.Linear(128, output_size)

        # Fully connected layer for further processing
        # self.fc = nn.Linear(input_size, 128)  # Adjust the output size as needed
        # self.agent_fc = nn.Linear(input_size['agent'], 128)
        # self.prev_fc = nn.Linear(input_size['prev'], 128)
        # self.target_fc = nn.Linear(input_size['target'], 128)
        # self.traffic_fc = nn.Linear(input_size['traffic'], 128)
        #
        # self.fc_combined_1 = nn.Linear(512, 256)
        # self.fc_combined_2 = nn.Linear(256, 128)  # Adjusted dimension
        # self.fc_combined = nn.Linear(128, output_size)

    def forward(self, x):
        flat_inputs = []
        for key, layer in zip(self.feature_keys, self.flat_layers):
            if isinstance(layer, nn.Embedding):
                flat_inputs.append(layer(x[key].long()))
            elif isinstance(layer, nn.Linear):
                flat_inputs.append(layer(x[key]))

        x = torch.cat(flat_inputs, dim=0)

        # Fully connected layer
        # Fully connected layer
        x = F.relu(self.fc(x))
        x = F.relu(self.fc_combined(x))
        # x = F.relu(self.fc_combined_2(x))
        x = F.softmax(self.fc_output(x), dim=0)

        return x
        #
        # x_agent = F.relu(self.agent_fc(x['agent']))
        # x_prev = F.relu(self.prev_fc(x['prev']))
        # x_target = F.relu(self.target_fc(x['target']))
        # x_traffic = F.relu(self.traffic_fc(x['traffic']))
        #
        # # Concatenate the processed inputs
        # x_combined = torch.cat([x_agent, x_prev, x_target, x_traffic], dim=-1)
        # x_combined_1 = F.relu(self.fc_combined_1(x_combined))
        # x_combined_2 = F.relu(self.fc_combined_2(x_combined_1))
        #
        # x = F.softmax(self.fc_combined(x_combined_2), dim=-1)
        #
        # return x


def train_ppo(env, epochs=50, batch_size=128, clip_ratio=0.2, gamma=0.99, learning_rate=0.001):
    obs_space = env.observation_space
    action_space = env.action_space.n

    # input_sizes = {key: obs_space[key].n if isinstance(obs_space[key], gym.spaces.Discrete) else obs_space[key].shape[0]
    #                for key in obs_space.spaces.keys()}
    policy_model = Policy(obs_space, action_space)
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        observations = {key: [] for key in obs_space.spaces.keys()}
        actions, rewards, old_probs = [], [[]], []
        for i in range(batch_size):
            obs, _ = env.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                obs = {key: torch.tensor(obs[key], dtype=torch.float32) for key in obs.keys()}
                action_probs = policy_model(obs)
                # print(obs, epoch)
                action_dist = Categorical(action_probs)
                action = action_dist.sample().item()

                for key in obs.keys():
                    observations[key].append(obs[key])

                actions.append(action)
                old_probs.append(action_dist.probs[action].item())

                obs, reward, terminated, truncated, _ = env.step(action)
                rewards[i].append(reward)

        returns, advantages, discounted_reward = [], [], 0

        for j in range(len(rewards)):
            for r in reversed(rewards[j]):
                discounted_reward = r + gamma * discounted_reward
                returns.insert(0, discounted_reward)

        baseline = np.mean(returns)
        advantages = np.array(returns) - baseline
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        optimizer.zero_grad()
        # print(len(rewards))
        print(rewards)
        for i in range(len(rewards)):
            obs_tensors = {key: observations[key][i].clone().detach() for key in observations.keys()}
            action_probs = policy_model(obs_tensors)
            chosen_probs = action_probs[actions[i]]
            ratio = chosen_probs / old_probs[i]

            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))

            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            surrogate = torch.min(ratio * advantages[i], clipped_ratio * advantages[i])
            loss = -surrogate + 0.01 * entropy  # Negative because we want to maximize the surrogate objective

            loss += 0.001 * sum(p.pow(2.0).sum() for p in policy_model.parameters())

            loss.backward()

        optimizer.step()
        print(f"Reward after {epoch + 1}: {np.mean(rewards)}")
        print("Training completed")
        print(f"Epoch {epoch + 1} completed")

    return policy_model


def main():
    """
    Creates the graph, makes the environment, and runs a randomized or human-controlled.

    :return:
    """
    graph, locations = make_grid_city(5, 5)
    env = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations)
    observation, info = env.reset()
    trained_policy = train_ppo(env)

    def test_policy(env, policy_model):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            obs_tensors = {key: torch.tensor(obs[key], dtype=torch.float32) for key in obs.keys()}
            action_probs = policy_model(obs_tensors)
            action = torch.argmax(action_probs).item()

            obs, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            # env.render()  # Optional: Uncomment this line if you want to visualize the environment

        print(f"Total Reward: {total_reward}")

    test_policy(env, trained_policy)

    # user_input = input("Manually move agent? (Y/N)\n")
    # if user_input.lower() == "y":
    #     env.render()
    #     env.manual_input_mode()
    # else:
    #     pass
    #     # for _ in range(1000):
    #     #     action = env.action_space.sample()  # agent policy that uses the observation and info
    #     #     observation, reward, terminated, truncated, info = env.step(action)
    #     #
    #     #     if terminated or truncated:
    #     #         observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
