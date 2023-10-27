from typing import SupportsFloat, Any

import gymnasium as gym
import pygame
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium import spaces
from random import randint
from gymnasium.envs.registration import register


class RoadNetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def render(self):
        return self._render_frame()

    def _render_frame(self):
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

        for i in range(self.n_nodes):
            for j in range(5):
                pygame.draw.line(canvas, (0, 0, 0), self.node_positions[i] * self.window_size, self.node_positions[self.graph[i][j]] * self.window_size)

        for i in range(self.n_nodes):
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                self.node_positions[i] * self.window_size,
                10
            )

        for i in range(self.n_nodes):
            if self._target_nodes[i]:
                pygame.draw.circle(
                    canvas,
                    (0, 127, 0),
                    self.node_positions[i] * self.window_size,
                    5
                )

        for i in range(self.n_nodes):
            if self._traffic_nodes[i]:
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0) if self._traffic_node_colours[i] == 1 else (255, 0, 0),
                    self.node_positions[i] * self.window_size,
                    12,
                    width=3
                )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
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
        return {
            "agent": int(self._agent_pos),
            "target": self._target_nodes,
            "prev": int(self._previous_pos)
        }

    def _get_info(self):
        return {}

    def step(self, action):
        self._timer += 1
        if self._timer % 10 == 0:
            self._traffic_node_colours = {index: np.random.randint(0, 2) > 0.5 for
                                          index in np.where(self._traffic_nodes)[0]}

        cur_pos = self._agent_pos
        self._agent_pos = self.graph[self._agent_pos][action]
        reward = 0.
        if self._target_nodes[self._agent_pos]:
            reward = 1.0
            self._target_nodes[self._agent_pos] = False
        elif self._agent_pos == cur_pos and action != 4:
            print(f"Tried going from {self.node_positions[cur_pos]} to "
                  f"{self.node_positions[self._agent_pos]}")
            reward = -5.0
        elif (self._traffic_nodes[cur_pos] and self._traffic_node_colours[cur_pos] == 0 and
              action != 4):
            reward = 5 * self._find_penalty(cur_pos)
            print(f"Tried going from {self.node_positions[cur_pos]} to "
                  f"{self.node_positions[self._agent_pos]} while "
                  f"red traffic light")

        print(reward)

        observation = self._get_obs()

        terminated = not np.any(self._target_nodes)

        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        self._previous_pos = cur_pos
        return observation, reward, terminated, False, info

    def _find_penalty(self, cur_pos):
        prev_node_coord = self.node_positions[self._previous_pos]
        cur_node_coord = self.node_positions[cur_pos]
        next_node_coord = self.node_positions[self._agent_pos]

        prev_vector = cur_node_coord - prev_node_coord
        next_vector = next_node_coord - cur_node_coord

        angle = np.arccos(np.clip(np.dot(prev_vector/np.linalg.norm(prev_vector),
                                         next_vector/np.linalg.norm(next_vector)), -1.0, 1.0))

        if 0 <= angle <= np.pi/4 or (315 * np.pi)/180 <= angle <= 2 * np.pi:
            direction_key = 0
        elif np.pi/4 <= angle <= (135 * np.pi)/180:
            direction_key = 1
        elif (135 * np.pi)/180 <= angle <= (225 * np.pi)/180:
            direction_key = 2
        else:
            direction_key = 3
        direction = self._angle_to_direction[direction_key]
        penalty = self._accident_prob[direction]

        return penalty


    def __init__(self, render_mode=None, graph=None, node_positions=None):
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

        self._agent_pos = randint(0, self.n_nodes - 1)
        self._target_nodes = np.random.rand(self.n_nodes) > 0.9
        self._traffic_nodes = np.random.rand(self.n_nodes) > 0.8
        while np.any(np.logical_and(self._traffic_nodes, self._target_nodes)):
            self._traffic_nodes = np.random.rand(self.n_nodes) > 0.8
        self._traffic_node_colours = {index: np.random.randint(0, 2) > 0.5 for
                                      index in np.where(self._traffic_nodes)[0]}
        self._timer = 1
        self._previous_pos = self._agent_pos
        self._angle_to_direction = {
            0: "straight",
            1: "left",
            2: "right",
            3: "back"
        }
        self._accident_prob = {
            "straight": -0.9,
            "left": -0.8,
            "right": -0.4,
            "back": -0.95
        }
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(self.n_nodes),
                "target": spaces.MultiBinary((self.n_nodes,)),
                "prev": spaces.Discrete(self.n_nodes)
            }
        )
        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 1024

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_pos = randint(0, self.n_nodes - 1)
        self._target_nodes = np.random.rand(self.n_nodes) > 0.9

        return self._get_obs(), self._get_info()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


register(
     id="RoadNetEnv-v0",
     entry_point="RoadNetEnv:RoadNetEnv",
)
