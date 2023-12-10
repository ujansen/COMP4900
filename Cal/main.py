import RoadNetEnv
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import actorcritic as ac


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



def t(x): return torch.from_numpy(x).float()

def main():
    """
    Creates the graph, makes the environment, and runs a randomized or human-controlled.

    :return:
    """
    graph, locations = make_grid_city(10,10)
    env = gym.make('RoadNetEnv-v0', render_mode=None, graph=graph, node_positions=locations)
    
    observation, info = env.reset()
    print(observation)
    
    
    episode_rewards = []
    observation= ac.featurizeObservation(observation)
    
    state_dim = len(observation)
    print(state_dim)
    n_actions = env.action_space.n
    actor = ac.Actor(state_dim, n_actions)
    critic = ac.Critic(state_dim)

    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.9
    memory = ac.Memory()
    max_steps = 200 

        
    for _ in range(500):
        done = False
        total_reward = 0
        state, info = env.reset()

        state= ac.featurizeObservation(state)
        steps = 0

        while not done:
            probs = actor(t(state))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            next_state, reward, done, idk,info = env.step(action.detach().data.numpy())
            next_state = ac.featurizeObservation(next_state)
            total_reward += reward
            
            memory.add(dist.log_prob(action), critic(t(state)), reward, done)
            
            state = next_state
            
            if done or (steps % max_steps == 0):
                last_q_val = critic(t(next_state)).detach().data.numpy()
                ac.train(memory, last_q_val,adam_actor,adam_critic,gamma)
                memory.clear()
        episode_rewards.append(total_reward)

    plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
    plt.title("Advantage actor critic, Rewards per episode")
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()







