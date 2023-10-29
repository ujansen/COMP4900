from random import randint
from math import degrees, atan2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.registration import register


class RoadNetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def render(self):
        """
        Renders the pygame window at every time step

        :return: Function call to render the window
        """
        return self._render_frame()

    def _render_frame(self):
        """
        Draws the pygame window at every time step with the relevant information

        :return: A numpy array representation of the pygame window
        """
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

        # draw the edges between nodes
        for i in range(self.n_nodes):
            for j in range(5):
                pygame.draw.line(canvas, (0, 0, 0), self.node_positions[i] * self.window_size,
                                 self.node_positions[self.graph[i][j]] * self.window_size)

        # draw the nodes
        for i in range(self.n_nodes):
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                self.node_positions[i] * self.window_size,
                10
            )

        # draw the targets
        for i in range(self.n_nodes):
            if self._target_nodes[i]:
                pygame.draw.circle(
                    canvas,
                    (255, 255, 0),
                    self.node_positions[i] * self.window_size,
                    5
                )

        # draw traffic lights (green or red)
        for i in range(self.n_nodes):
            if self._traffic_nodes[i]:
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0) if self._traffic_node_colours[i] == 1 else (255, 0, 0),
                    self.node_positions[i] * self.window_size,
                    12,
                    width=3
                )

        # draw the agent
        pygame.draw.circle(
            canvas,
            (0, 229, 255),
            self.node_positions[self._agent_pos] * self.window_size,
            8
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def _get_obs(self):
        """
        Returns the observation space

        :return: A dictionary defining the current state of the environment
        """
        return {
            "agent": int(self._agent_pos),
            "target": self._target_nodes,
            "prev": int(self._previous_pos),
            "traffic": self._traffic_node_colours[self._agent_pos] if self._agent_pos in self._traffic_node_colours
            else 1
        }

    def _get_info(self):
        """
        Auxiliary function for debugging

        :return: Relevant information for debugging
        """
        return {}

    def step(self, action):
        """
        Specifics of the next step taken by the agent in the environment.
        This step leads to a new state, and rewards, and could also lead to termination.
        This function calculates the aforementioned new state and reward depending on the chosen action

        :param action: The action to be taken by the agent in the current state
        :return: observation: the new state, reward: reward received after taking the current action,
                 terminated: if the episode has terminated or not, truncated: if memory has lapsed,
                 info: debugging information.
        """
        # toggle the traffic lights from green to red and vice versa every 10 time steps
        if self._timer % 10 == 0:
            self._traffic_node_colours = {key: 1 - value for key, value in self._traffic_node_colours.items()}

        terminated = False
        cur_pos = self._agent_pos
        self._agent_pos = self.graph[self._agent_pos][action]
        reward = 0.

        # check if the agent reaches target
        if self._target_nodes[self._agent_pos]:
            reward = 1.0
            self._target_nodes[self._agent_pos] = False

        # check if the agent tries to move out of bounds
        elif self._agent_pos == cur_pos and action != 4:
            print("Tried going out of bounds")
            print(f"Tried going from {self.node_positions[cur_pos]} to "
                  f"{self.node_positions[self._agent_pos]}")
            reward = -1.0
            terminated = True

        # check if the agent runs a traffic light
        elif (self._traffic_nodes[cur_pos] and self._traffic_node_colours[cur_pos] == 0 and
              action != 4):
            if np.random.rand() < self._find_accident_prob(cur_pos):
                reward = -5.0
                terminated = True
            print(f"Tried going from {self.node_positions[cur_pos]} to "
                  f"{self.node_positions[self._agent_pos]} while "
                  f"red traffic light")

        # check if the agent tries taking a U-turn
        elif self._agent_pos == self._previous_pos and self._timer > 1:
            print(f"Tried taking a u-turn from {self.node_positions[cur_pos]} to "
                  f"{self.node_positions[self._agent_pos]} and timer is {self._timer}")
            if np.random.rand() > 0.75:
                reward = -2.0
                terminated = True

        if reward != 0:
            print(reward)

        observation = self._get_obs()

        # if not terminated already, terminate after all target nodes have been reached
        terminated = not np.any(self._target_nodes) if not terminated else terminated

        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        # update previous position to current position only if action led to new node
        self._previous_pos = cur_pos if self._agent_pos != cur_pos else self._previous_pos
        self._timer += 1
        return observation, reward, terminated, False, info

    def _find_accident_prob(self, cur_pos):
        """
        A function that returns the probability of an accident occurring
        if the agent decides to run a red light.
        First, the direction is calculated depending on the agent's previous position,
        current position, and next position using vector algebra.
        Following this, a lookup table is used to calculate the probability an accident will occur.

        :param cur_pos: The current position of the agent before deciding
                        to move (i.e., the position of the traffic light agent is at)
        :return: A probability of an accident occurring
        """
        prev_node_coord = self.node_positions[self._previous_pos]
        cur_node_coord = self.node_positions[cur_pos]
        next_node_coord = self.node_positions[self._agent_pos]

        prev_vector = cur_node_coord - prev_node_coord
        next_vector = next_node_coord - cur_node_coord
        # prev_vector[1] *= -1 if prev_vector[1] != 0 else prev_vector[1]
        # next_vector[1] *= -1 if next_vector[1] != 0 else next_vector[1]

        # find the angle between vector from prev node to cur node and cur node to next node
        # angle = np.arccos(np.clip(np.dot(prev_vector / np.linalg.norm(prev_vector),
        #                                  next_vector / np.linalg.norm(next_vector)), -1.0, 1.0))
        angle = np.rad2deg(np.arctan2(np.cross(prev_vector, next_vector), np.dot(prev_vector, next_vector)))
        print(prev_node_coord, cur_node_coord, next_node_coord)
        print(angle, prev_vector, next_vector)

        # if angle between 0 and 45 or 315 and 360, snap to the straight direction
        if 0 <= angle < 90:
            direction_key = 0

        # if angle between 45 and 135, snap to the left direction
        elif -90 <= angle < 0:
            direction_key = 1

        # if angle between 135 and 225, snap to the right direction
        elif 90 <= angle < 180:
            direction_key = 2

        # snap to U-turn or the back direction
        else:
            direction_key = 3
        direction = self._angle_to_direction[direction_key]
        print(direction)

        # find the probability of accident depending on the direction
        probability = self._accident_prob[direction]

        return probability

    def __init__(self, render_mode=None, graph=None, node_positions=None):
        """
        Initializes the environment with all necessary information

        :param render_mode: If render mode is human or rgb_array.
        :param graph: The graph which acts as the environment.
        :param node_positions: The positions of each node in the graph as an (x, y) coordinate.
        """
        # if no graph is specified, create a random graph
        if graph is None:
            self.n_nodes = 100
            self.graph = np.zeros((self.n_nodes, 5), dtype=int)
            for i in range(self.n_nodes):
                self.graph[i][0] = randint(0, self.n_nodes - 1)
                self.graph[i][1] = randint(0, self.n_nodes - 1)
                self.graph[i][2] = randint(0, self.n_nodes - 1)
                self.graph[i][3] = randint(0, self.n_nodes - 1)
                self.graph[i][4] = i
            self.node_positions = np.random.rand(self.n_nodes, 2)
        else:
            self.n_nodes = graph.shape[0]
            self.node_positions = node_positions
            self.graph = graph

        # randomly set the agent, target, and traffic locations (make sure traffic, agent, and targets do not overlap)
        self._agent_pos = randint(0, self.n_nodes - 1)
        self._target_nodes = np.random.rand(self.n_nodes) > 0.9
        while self._target_nodes[self._agent_pos] and not np.any(self._target_nodes):
            self._target_nodes = np.random.rand(self.n_nodes) > 0.9

        self._traffic_nodes = np.random.rand(self.n_nodes) > 0.6
        while np.any(np.logical_and(self._traffic_nodes, self._target_nodes)) or self._traffic_nodes[self._agent_pos]:
            self._traffic_nodes = np.random.rand(self.n_nodes) > 0.6

        # randomly set each traffic light to red or green (each has a 50% probability of being either)
        self._traffic_node_colours = {index: np.random.randint(0, 2) > 0.5 for
                                      index in np.where(self._traffic_nodes)[0]}

        # set a timer that toggles the traffic lights
        self._timer = 1
        self._previous_pos = self._agent_pos

        # a dictionary to snap the angle to a direction
        self._angle_to_direction = {
            0: "straight",
            1: "left",
            2: "right",
            3: "back"
        }

        # a dictionary that enumerates the probabilities of accidents while running
        # a red light
        self._accident_prob = {
            "straight": 0.9,
            "left": 0.8,
            "right": 0.4,
            "back": 0.95
        }

        # define the observation space
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(self.n_nodes),
                "target": spaces.MultiBinary((self.n_nodes,)),
                "prev": spaces.Discrete(self.n_nodes),
                "traffic": spaces.Discrete(2)
            }
        )
        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 700

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment with all necessary parameters

        :param seed: To control randomized resetting
        :param options: Auxiliary information
        :return: Initial state, debugging info if relevant
        """
        super().reset(seed=seed)
        self._agent_pos = randint(0, self.n_nodes - 1)
        self._target_nodes = np.random.rand(self.n_nodes) > 0.9
        while self._target_nodes[self._agent_pos] and not np.any(self._target_nodes):
            self._target_nodes = np.random.rand(self.n_nodes) > 0.9
            
        self._traffic_nodes = np.random.rand(self.n_nodes) > 0.6
        while np.any(np.logical_and(self._traffic_nodes, self._target_nodes)) or self._traffic_nodes[self._agent_pos]:
            self._traffic_nodes = np.random.rand(self.n_nodes) > 0.6

        self._traffic_node_colours = {index: np.random.randint(0, 2) > 0.5 for
                                      index in np.where(self._traffic_nodes)[0]}
        self._timer = 1
        self._previous_pos = self._agent_pos

        return self._get_obs(), self._get_info()

    def close(self):
        """
        Closes the environment

        :return:
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def manual_input_mode(self):
        """
        Human-controlled agent for the environment.

        :return:
        """
        run = True
        while run:
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 3
                    elif event.key == pygame.K_LEFT:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 2
                    elif event.key == pygame.K_SPACE:
                        action = 4

            if action is not None:
                observation, reward, terminated, truncated, info = self.step(action)
                if terminated or truncated:
                    observation, info = self.reset()
            self.render()


register(
    id="RoadNetEnv-v0",
    entry_point="RoadNetEnv:RoadNetEnv",
)
