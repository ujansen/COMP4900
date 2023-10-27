import gymnasium as gym
import pygame
import numpy as np
from data_structures import *
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium import spaces
from random import randint
from gymnasium.envs.registration import register

# Helper class specifically for rewards logic. 
# TODO - Consider sampling rewards from a distribution
class Rewards():
    # Add more stuff accordingly!
    TARGET_PICKED_UP = 10
    ATTEMPTED_MOVE_OUT_OF_BOUNDS = -300
    MOVED_AT_RED_STOPLIGHT = -100
    MOVED_WRONG_WAY_ON_DIRECTED_ROAD = -200

class RoadNetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Add accordingly!
    # REWARDS = {
    #     "target_picked_up": 10,
    #     "attempted_out_of_bounds_move": -100,
    #     "moved_at_red_stoplight": -100,
    #     "moved_wrong_way_on_directed_road": -100,
    # }

    def __init__(self, render_mode=None, graph: Graph = None, targets_freq = 0.1):
        if graph is None:
            raise Exception("Graph must be provided. TODO - implement random graphs")
            # Code below needs to be updated to new structure. But may not even be needed
            # self.n_nodes = 100
            # self.graph = np.zeros((self.n_nodes, 5), dtype=int)
            # for i in range(self.n_nodes):
            #     self.graph[i][0] = randint(0, self.n_nodes - 1)
            #     self.graph[i][1] = randint(0, self.n_nodes - 1)
            #     self.graph[i][2] = randint(0, self.n_nodes - 1)
            #     self.graph[i][3] = randint(0, self.n_nodes - 1)
            #     self.graph[i][4] = i
            # self.node_positions = np.random.rand(self.n_nodes, 2)
        else:
            self.graph = graph
            self.targets_freq = 0.1            
            self.n_nodes = graph.num_nodes() 
            
        self._agent_pos = randint(0, self.n_nodes - 1)
        self._is_target_node_pos = np.random.rand(self.n_nodes) < targets_freq

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(self.n_nodes),
                "target": spaces.MultiBinary((self.n_nodes,)),
            }
        )
        self.action_space = spaces.Discrete(5)

        self._action_to_direction = {0: "center", 1: "left", 2: "right", 3: "up", 4: "down"}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 1024 * 0.7  # ** Times 0.7 since otherwise its too big for my computer... **

        self.window = None
        self.clock = None
    
    def render(self):
        return self._render_frame()

    def _render_frame(self):
        """ Render the current environment. """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw a circle to represent each node
        for i in range(self.n_nodes):
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                self.graph.nodes[i].location * self.window_size,
                10
            )

        # For each node, draw a line (edge) from it to each of its out nodes 
        for i in range(self.n_nodes):
            cur_node = self.graph.nodes[i]
            for direction in DIRECTIONS:
                other_node_index = cur_node.out_nodes[direction]
                # If no other node in the given direction, no line to draw
                if direction == "center" or other_node_index is None:
                    continue

                pygame.draw.line(canvas, (0, 0, 0), cur_node.location * self.window_size, 
                                 self.graph.nodes[other_node_index].location * self.window_size)

        # Color the target nodes 
        for i in range(self.n_nodes):
            if self._is_target_node_pos[i]:
                pygame.draw.circle(
                    canvas,
                    (0, 127, 0),
                    self.graph.nodes[i].location * self.window_size,
                    5
                )

        # Color the stop light nodes based on their current stop light colours
        for i in range(self.n_nodes):
            cur_node = self.graph.nodes[i]
            if not isinstance(cur_node, StoplightNode):
                continue
            pygame.draw.circle(
                canvas,
                (0, 255, 0) if cur_node.is_green else (255, 0, 0),
                self.graph.nodes[i].location * self.window_size,
                12,
                width=3
            )

        # Colour the node at which the agent currently resides
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self.graph.nodes[self._agent_pos].location * self.window_size,
            8
        )

        # TODO - draw a '>' pointed in the right direction , for all directed nodes 
        # i.e nodes with a directed street!

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    # ** TODO - INCOMPLETE. Really think about what all the state info should be **
    # Need more stuff here. e.g:
    # -cur node a red traffic light?
    # -what are the valid directions I can take (but not just legal moves.) 
    #  e.g I need to know if moving up is allowed or if that is like crashing into a wall
    # -what is the required direction of traffic (i.e are we a one way street. o.w any direction allowed)
    def _get_obs(self):
        """
        Returns all relevant observations of the current environment. These observations constitute 
        the current state.
        """
        # *** INCOMPLETE ***
        return {
            "agent": int(self._agent_pos),
            # ** Should this not be a matrix? otherwise how will the agent know what L/R/Up/Down?
            # that would depend on the dimensions of the grid, which is unknown. So basically,
            # having the representation as a grid seems best!
            "target": self._is_target_node_pos
        }
    
    # ? Nani ?
    def _get_info(self):
        return {}


    def step(self, action):
        """
        Update the environment based on the action the agent has just taken, and return:
        (observation, reward, terminated, truncated, info)
        """
        reward = self.compute_reward(action)
        self.update_env(action)

        # Observations of the now updated environment to return to the user
        observation = self._get_obs()

        terminated = not np.any(self._is_target_node_pos)

        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info
    
    # Where the magic happens
    # TODO - consider adding probabilities. e.g maybe 50% chance of a negative reward being given
    #        if you move at a stoplight (tsk tsk)
    def compute_reward(self, action):
        """
        Compute reward given the action the agent will take.
        """
        # The reward at a current step can be a sum of sub-rewards. e.g negative reward for skipping
        # red light, and positive reward since next node is a target node.
        net_reward = 0
    
        cur_node = self.graph.nodes[self._agent_pos]
        action_direction = self._action_to_direction[action]
        next_node_index = cur_node.out_nodes[action_direction]

        # Case where agent attempted to move out of bounds
        if next_node_index is None:
            net_reward += Rewards.ATTEMPTED_MOVE_OUT_OF_BOUNDS
        
        # Case where agent moved when at a red stoplight
        if isinstance(cur_node, StoplightNode) and not cur_node.is_green:
            net_reward += Rewards.MOVED_AT_RED_STOPLIGHT

        # Case where agent moves in the wrong direction on a one-way street.
        # For now, staying in the same spot is considered fine, even if not at a red stoplight.
        if cur_node.directed and cur_node.directed != action_direction and action_direction != "center":
            net_reward += Rewards.MOVED_WRONG_WAY_ON_DIRECTED_ROAD

        # Case where next node moved to is a target
        if next_node_index is not None and self._is_target_node_pos[next_node_index]:
            net_reward += Rewards.TARGET_PICKED_UP

        return net_reward
    

    def update_env(self, action):
        """
        Update the environment based on the agent's action. Also update based on the passage of one
        timestep (e.g for traffic lights).
        """
        # Update the position of the agent.
        # next_node_index None occurs if agent tries to move in an invalid direction. In that case, 
        # agent stays at current position. Otherwise moves to new position.
        cur_node = self.graph.nodes[self._agent_pos]
        next_node_index = cur_node.out_nodes[self._action_to_direction[action]]
        if next_node_index is not None:
            self._agent_pos = next_node_index 

        # If agent landed on target, remove target from grid.
        landed_on_target = self._is_target_node_pos[self._agent_pos]
        if landed_on_target:
            self._is_target_node_pos[self._agent_pos] = False

        # Update stoplight nodes as a timestep has passed
        for node in self.graph.nodes:
            if isinstance(node, StoplightNode):
                node.simulate_next_timestep()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._is_target_node_pos = np.random.rand(self.n_nodes) < self.targets_freq

        # Ensure the starting agent pos is not on a target
        while True:
            self._agent_pos = randint(0, self.n_nodes - 1)
            if not self._is_target_node_pos[self._agent_pos]:
                break

        return self._get_obs(), self._get_info()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


register(
     id="RoadNetEnv-v0",
     entry_point="RoadNetEnv:RoadNetEnv",
)
