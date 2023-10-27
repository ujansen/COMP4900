import random
import numpy as np
import gymnasium as gym
import RoadNetEnv
from data_structures import *

# NEW version - makes use of the required data structures. Note that all roads created are undirected here!
# TODO - `Implement example_city_with_one_way_Streets()` below .
def make_grid_city(dim_x, dim_y, stoplight_freq = 0.1):
    graph = Graph()

    # Create all the nodes making up the graph. Random chance of stop light nodes based on freq value provided.
    n_nodes = dim_x * dim_y
    graph.nodes = [StoplightNode(i) if random.random() < stoplight_freq else Node(i) for i in range(n_nodes)]

    # Used in computing node locations below
    offset_per_x = 1.0 / dim_x
    offset_per_y = 1.0 / dim_y
    
    # Initialize all of the nodes, one at a time, by iterating across a grid of indices corresponding
    # to the provided dimensions.
    for x in range(dim_x):
        for y in range(dim_y):
            index = y * dim_x + x
            cur_node = graph.nodes[index]
            print(cur_node)

            # Location must be an np.array since multiplied against in other functions
            cur_node.location = np.array(
                [x * offset_per_x + (offset_per_x / 2),  y * offset_per_y + (offset_per_y / 2)])
           
            # Init out_nodes
            # 'None' means an attempt to move in that direction won't work (i.e at edge here), so agent would stay
            # in same spot after such an attempt.
            cur_node.out_nodes["center"] = index 
            cur_node.out_nodes["left"] = y * dim_x + (x - 1) if x > 0 else None
            cur_node.out_nodes["right"] = y * dim_x + (x + 1) if x < dim_x - 1 else None
            cur_node.out_nodes["up"] = (y - 1) * dim_x + x if y > 0 else None
            cur_node.out_nodes["down"] = (y + 1) * dim_x + x if y < dim_y - 1 else None

            print("cur node index:")
            print(cur_node.index)
            print("Out nodes:")
            print(cur_node.out_nodes)

    return graph

# OLD version !
# def make_grid_city(dim_x, dim_y):
#     # For each node, stores all possible nodes it can transition to
#     # i.e graph[i] = [left node, up node,  right node, down node, current node]
#     graph = np.zeros((dim_x * dim_y, 5), dtype=int)

#     # For each node, stores its relative location in the entire grid, as a proportion of the 
#     # way the node is to the right most of the map, and the bottom most of the map, respectively.
#     locations = np.zeros((dim_x * dim_y, 2), dtype=
#                          float)
#     offset_per_x = 1.0 / dim_x
#     offset_per_y = 1.0 / dim_y
#     # Iterate across grid, going across one column at a time (top to bottom)
#     for x in range(dim_x):
#         for y in range(dim_y):
#             index = y * dim_x + x
#             print(index)

#             locations[index][0] = x * offset_per_x + (offset_per_x / 2)
#             locations[index][1] = y * offset_per_y + (offset_per_y / 2)

#             # Left node
#             if x > 0:
#                 graph[index][0] = y * dim_x + (x - 1)
#             else:
#                 graph[index][0] = index

#             # Up node 
#             if y > 0:
#                 graph[index][1] = (y - 1) * dim_x + x
#             else:
#                 graph[index][1] = index

#             # Right node
#             if x < dim_x - 1:
#                 graph[index][2] = y * dim_x + (x + 1)
#             else:
#                 graph[index][2] = index

#             # Down node
#             if y < dim_y - 1:
#                 graph[index][3] = (y + 1) * dim_x + x
#             else:
#                 graph[index][3] = index

#             # Center (current) node
#             graph[index][4] = index

#     return graph, locations


# TODO - we have everything we need logically to make one way streets.
# Lets make an example city with one way streets, hard coded for now.
# i.e start with grid city and overwrite 'None' directed values with an actual direction, for nodes
# of interest!
# Will also require adjusting the render code accordingly!
def example_city_with_one_way_Streets():
    graph, locations = make_grid_city(5,5)
    return graph, locations
