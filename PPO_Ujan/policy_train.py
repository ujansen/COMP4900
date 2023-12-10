import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import torch.nn.functional as F
from main import make_grid_city
import matplotlib.pyplot as plt
import RoadNetEnv


# Define the policy and value networks
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return self.fc4(x)

# PPO Algorithm
class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, clip_ratio, explore_exploit_epsilon,
                 value_coeff, entropy_coeff, gae_lambda, max_grad_norm):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = PolicyNetwork(state_dim, action_dim).cuda()  # Move policy_net to GPU
        self.value_net = ValueNetwork(state_dim).cuda()  # Move value_net to GPU
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.explore_exploit_epsilon = explore_exploit_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.losses, self.average_rewards = [], []

    def compute_advantages(self, rewards, values, terminateds):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        advantage = 0
        next_value = values[-1]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - terminateds[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - terminateds[t])
            advantages[t] = advantage
            next_value = values[t]

        return advantages

    def update(self, states, actions, old_probs, advantages, returns):
        states = torch.tensor(states, dtype=torch.float32).cuda()  # Move states to GPU
        actions = torch.tensor(actions, dtype=torch.int64).cuda()  # Move actions to GPU
        old_probs = torch.tensor(old_probs, dtype=torch.float32).cuda()  # Move old_probs to GPU
        advantages = torch.tensor(advantages, dtype=torch.float32).cuda()  # Move advantages to GPU
        returns = torch.tensor(returns, dtype=torch.float32).cuda()  # Move returns to GPU

        # Compute new action probabilities and values
        new_probs = self.policy_net(states).gather(1, actions.unsqueeze(1))
        values = self.value_net(states).view(-1)

        # PPO loss
        ratio = new_probs / (old_probs + 1e-5)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = -torch.sum(new_probs * torch.log(new_probs.clamp(min=1e-8)), dim=-1).mean()
        policy_loss -= self.entropy_coeff * entropy

        # Value loss
        value_loss = nn.MSELoss()(values, returns)

        # Total loss
        loss = policy_loss + self.value_coeff * value_loss
        self.losses.append(loss.item())
        self.average_rewards.append(np.mean(returns.cpu().numpy()))  # Move to CPU for logging

        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()


# Training loop
def train_ppo_batch(agent, env, epochs, batch_size):
    for epoch in range(epochs):
        all_states, all_actions, all_rewards, all_old_probs, all_terminated = [], [], [], [], []
        all_values, all_returns = [], []

        # Collect experiences from multiple parallel environments
        for _ in range(batch_size):
            states, actions, rewards, old_probs, terminateds = [], [], [], [], []
            values = []

            state, _ = env.reset()

            while True:
                with torch.no_grad():
                    flat_state = np.concatenate([np.atleast_1d(state['agent']), np.atleast_1d(state['target']),
                                                 np.atleast_1d(state['prev']),
                                                 np.atleast_1d(state['traffic']),
                                                 state['direction']])

                    old_action_probs = agent.policy_net(torch.FloatTensor(flat_state).cuda())
                    value = agent.value_net(torch.FloatTensor(flat_state).cuda())

                # if np.random.rand() < agent.explore_exploit_epsilon:
                #     action = np.random.choice(agent.action_dim)
                # else:
                #     action = np.argmax(old_action_probs.cpu().numpy())
                action = np.random.choice(range(agent.action_dim), p=old_action_probs.cpu().numpy())
                next_state, reward, terminated, truncated, _ = env.step(action)

                # Update lists for the current environment
                states.append(flat_state)
                actions.append(action)
                rewards.append(reward)
                old_probs.append(old_action_probs[action].item())
                terminateds.append(terminated)
                values.append(value.item())

                state = next_state

                if terminated or truncated:
                    break

            # Update lists for all environments
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_old_probs.extend(old_probs)
            all_terminated.extend(terminateds)
            all_values.extend(values)

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = agent.compute_advantages(all_rewards, all_values, all_terminated)
        returns = advantages + np.array(all_values)

        # Normalize advantages
        # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8) if len(advantages) > 1 \
        #     else advantages
        # returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8) if len(returns) > 1 \
        #     else returns

        # Update policy and value networks using batch data
        agent.update(all_states, all_actions, all_old_probs, advantages, returns)

        plot_training_curve(agent.average_rewards, agent.losses)


def plot_training_curve(average_rewards, losses):
    # episodes = np.arange(1, len(average_rewards) + 1)
    print(average_rewards, losses)
    # Plot Average Reward
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(episodes, average_rewards)
    # plt.title('Average Reward per Episode')
    # plt.xlabel('Episodes')
    # plt.ylabel('Average Reward')
    #
    # # Plot Total Loss
    # plt.subplot(1, 2, 2)
    # plt.plot(episodes, losses, label='Total Loss')
    # plt.title('Total Loss per Episode')
    # plt.xlabel('Episodes')
    # plt.ylabel('Total Loss')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()


gamma = 0.99
clip_ratio = 0.4
explore_exploit_epsilon = 0.01
value_coeff = 0.5
entropy_coeff = 0.
gae_lambda = 0.98
epochs = 400
max_grad_norm = 0.6
lr = 0.0001

# Example usage
graph, locations = make_grid_city(5, 5)
env = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations)
# env = gym.make('RoadNetEnv-v0')

state_dim = 8
action_dim = 5

ppo_agent = PPO(state_dim,
                action_dim,
                lr=lr,
                gamma=gamma,
                clip_ratio=clip_ratio,
                explore_exploit_epsilon=explore_exploit_epsilon,
                value_coeff=value_coeff,
                entropy_coeff=entropy_coeff,
                gae_lambda=gae_lambda,
                max_grad_norm=max_grad_norm)

# Batch training loop
train_ppo_batch(ppo_agent, env, epochs=100, batch_size=64)

# Save the trained model
torch.save(ppo_agent.policy_net.state_dict(), "ppo_policy_batch.pt")
torch.save(ppo_agent.value_net.state_dict(), "ppo_value_batch.pt")
