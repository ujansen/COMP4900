import gymnasium as gym
import numpy as np


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


def main():
    """
    Creates the graph, makes the environment, and runs a randomized agent that samples an action
    from the action space.

    :return:
    """
    graph, locations = make_grid_city(10, 10)
    env = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations)
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
