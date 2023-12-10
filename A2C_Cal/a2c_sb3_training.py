import RoadNetEnv
import gymnasium as gym
from stable_baselines3 import A2C
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




graph, locations = make_grid_city(5,5)

# SINGLE TARGET TRAINING

# env = gym.make('RoadNetEnv-v0', render_mode=None, graph=graph, node_positions=locations, multi_target=False)    
# model = A2C("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)
# model.save("a2c_RoadNetEnv_single_target")

# MULTIPLE TARGET TRAINING

# env = gym.make('RoadNetEnv-v0', render_mode=None, graph=graph, node_positions=locations, multi_target=True)    
# model = A2C("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)
# model.save("a2c_RoadNetEnv_multi_target")