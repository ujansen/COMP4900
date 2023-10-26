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
            "target": self._target_nodes
        }

    def _get_info(self):
        return {}

    def step(self, action):
        self._agent_pos = self.graph[self._agent_pos][action]

        if self._target_nodes[self._agent_pos]:
            reward = 1.0
            self._target_nodes[self._agent_pos] = False
        else:
            reward = 0.0

        observation = self._get_obs()

        terminated = not np.any(self._target_nodes)

        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

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

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(self.n_nodes),
                "target": spaces.MultiBinary((self.n_nodes,)),
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
