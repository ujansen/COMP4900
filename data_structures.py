"""Objects of the classes defined below are used in defining an environment."""
import random
import numpy as np
from typing import TypedDict, Optional, List, Literal
from dataclasses import dataclass

#  Directions the agent can attempt to take (center refers to staying in the same spot)
DIRECTIONS = ["center", "left", "right", "up", "down"]

"""
Each node has five outnodes it can go to, which correspond to staying at the same node, going left, right, up,
or down. If movement in a particular direction is not possible, then we assign None
(e.g if there is a wall to the left or no road below, etc.)
"""
class OutNodes(TypedDict):
    center: Optional[int]
    left: Optional[int]
    right: Optional[int]
    up: Optional[int]
    down: Optional[int]


class Node:    
    def __init__(
        self,
        index: int = None, 
        location: np.ndarray = None,
        out_nodes: OutNodes = None,
        directed: Optional[Literal["center", "left", "right", "up", "down"]] = None,
    ) -> None:
        """
        index - Index of the node in the corresponding Graph's node list.
        location - location of the node in the grid, as a proportion of the way the node is to the right most of the grid
            and the bottom most of the grid, respectively. ** only makes sense for a grid, not a generalized graph **
        out_nodes - all nodes that can be reached from the current node (each out node represents an out edge), 
            whether legally or illegally.
        directed - Indicates the direction if dealing with a one way street. If None then any direction
            is legal.
        """
        self.index = index
        self.location = location
        self.out_nodes = out_nodes
        self.directed = directed
        
        if out_nodes is not None:
            self.out_nodes = out_nodes
        # ** Note: cannot set default value in the param for init. Otherwise dict is shared across all created nodes **
        else:
            self.out_nodes = {
                "center": None,
                "left": None,
                "right": None,
                "down": None
            }

class StoplightNode(Node):
    
    def __init__(
        self,
        index: int = None, 
        location: tuple[int,int] = None,
        out_nodes: OutNodes = None, 
        green_to_red_prob = 0.1, 
        red_to_green_prob = 0.2, 
        start_green = None,
        directed: Optional[Literal["center", "left", "right", "up", "down"]] = None,
    ) -> None:
        """
        is_green - If true, it means it is legal to move through the intersection. Otherwise,
            it is a red light and movement is illegal. No yellow lights. May wish to incorporate later.
        green_to_red - Probability a green light switches to red in the next time step
        red_to_green_prob - Probability a red light switches to green in the next time step

        Note: May later wish to change the logic of how the lights switch.
        """
        super().__init__(index, location, out_nodes, directed)

        self.green_to_red_prob = green_to_red_prob
        self.red_to_green_prob = red_to_green_prob

        if start_green is not None:
            self.is_green = start_green
        else:
            self.is_green = 1 == random.randint(0,1) # 50/50 chance of starting green or red

    def simulate_next_timestep(self):
        """
        Simulates passage of one time step for the stoplight. Should be called in every timestep. 
        The update may keep the light colour the same or toggle it.
        """
        # There is some probability the light colour will toggle 
        if ((self.is_green and random.random() < self.green_to_red_prob) or 
            (not self.is_green and random.random() < self.red_to_green_prob)): 
                self.is_green = not self.is_green


class Graph:
    """
    We define the graph representation of the environment to be an ordered list of nodes.
    """
    def __init__(self, nodes: list[Node] = []) -> None:
        self.nodes = nodes
    
    def num_nodes(self):
        return len(self.nodes)


