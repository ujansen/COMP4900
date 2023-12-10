# This is trying to reduce the state space as much as it feasibly can.
# First, it simply generates a single cycle. Then, the agent can find the shortest path to get on to that cycle.
# Next, it discards any non-goal nodes, so that the task is simply to cross all nodes.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def has_index_cycle(arr):
    n = arr.size
    visited = np.zeros(n, dtype=bool)

    for i in arr:
        if i < 0 or i >= n or visited[i]:
            return False
        visited[i] = True

    return np.all(visited)


def train(n_nodes: int, n_episodes: int):
    values = np.random.rand(n_nodes, n_nodes) * 0.01
    epsilon = 0.2
    gamma = 0.95
    lr = 0.1

    positions_for_placeholder_cost = np.random.rand(n_nodes, 2)

    rewards = []

    # we iterate over episodes
    for e in range(n_episodes):
        state = 0
        done = False
        total_reward = 0
        not_visited = np.ones((n_nodes,), dtype=int)
        not_visited[0] = 0
        visited_all = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(n_nodes)
            else:
                action = np.argmax(values[state, :])

            next_state = action
            # reward = 5 * not_visited[next_state] - np.linalg.norm(
            #    positions_for_placeholder_cost[next_state] - positions_for_placeholder_cost[state]
            # )
            reward = 5 * not_visited[next_state] - np.linalg.norm(
                positions_for_placeholder_cost[next_state] - positions_for_placeholder_cost[state]
            )
            not_visited[next_state] = 0
            if not visited_all and np.sum(not_visited) == 0:
                visited_all = True

            if visited_all and next_state == 0:
                done = True

            values[state, action] = (1 - lr) * values[state, action] + lr * (reward + gamma * np.max(values[next_state, :]))
            total_reward = total_reward + reward

            state = next_state

        rewards.append(total_reward)
    print(rewards)
    print(max(rewards))

    naive_policy = np.argmax(values, axis=1)
    print(naive_policy)

    # We can't really do a standard argmax because the policy NEEDS to be a complete cycle.
    # We could solve using a travelling salesman algorithm, but that'd just be weird.
    # I'll just take the highest non-visited one sequentially.
    # This should usually produce decent results.
    policy = np.zeros((n_nodes,), dtype=int)
    visited = np.ones((n_nodes, ), dtype=float)
    for i in range(n_nodes):
        temp_values = values[i] + visited
        target = np.argmax(temp_values)
        policy[i] = target
        visited[target] = np.NINF

    return values, policy, positions_for_placeholder_cost


def nearest_neighbour(positions: np.array):
    state = 0
    route = [0]
    visited = np.zeros((positions.shape[0],), dtype=int)

    while np.sum(visited) < visited.shape[0]:
        distances = np.linalg.norm(positions - positions[state], axis=1) + visited * 100000
        state = np.argmin(distances)
        visited[state] = 1
        route.append(state)
    return np.array(route)


def ant_colony(positions: np.array, num_ants: int):
    n_nodes = positions.shape[0]
    edge_weights = np.ones((n_nodes, n_nodes), dtype=float)
    for ant in range(num_ants):
        state = 0
        not_visited = np.ones((n_nodes,), dtype=int)
        path = [state]
        while np.sum(not_visited) != 0:
            indices = np.argwhere(not_visited == 1)
            indices = indices.reshape((indices.shape[0],))
            weights = edge_weights[state][indices]
            probabilities = weights / np.sum(weights)
            state = np.random.choice(indices, p=probabilities)
            not_visited[state] = 0
            path.append(state)
        path = np.array(path)
        differences = np.diff(positions[path], axis=0)
        distances = np.linalg.norm(differences, axis=1)
        cycle_length = np.sum(distances)
        pheromone = 1 / cycle_length
        rolled_path = np.roll(path, 1)
        for i in range(n_nodes):
            edge_weights[path[i]][rolled_path[i]] += pheromone

    policy = np.zeros((n_nodes,), dtype=int)
    visited = np.ones((n_nodes,), dtype=float)
    for i in range(n_nodes):
        temp_values = edge_weights[i] * visited
        target = np.argmax(temp_values)
        policy[i] = target
        visited[target] = 0

    return policy


if __name__ == "__main__":
    n_nodes = 10000
    #points = np.random.rand(n_nodes, 2)
    #values, policy, positions = train(n_nodes, 1000)
    #print(policy)
    #print(has_index_cycle(policy))

    # plt.plot(*positions[policy].T)
    positions = np.random.rand(n_nodes, 2)
    plt.plot(*positions[nearest_neighbour(positions)].T)
    plt.show()

    #plt.plot(*positions[ant_colony(positions, 1000)].T)
    #plt.show()
