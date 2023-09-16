import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class UrbanCitySetting(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512
        self.direction = np.random.randint(1, 4)
        self.observation_space = spaces.Discrete(11)
        self.action_space = spaces.Discrete(4)  # [straight, left, right, stop]
        self._agent_location = np.array([0, 0])
        self._target_location = np.array([size - 1, size - 1])

        self._obstacle_locations = [np.array([np.random.randint(1, self.size - 2),
                                              np.random.randint(1, self.size - 2)])
                                    for _ in range(10)]
        self._intersections = self._intersections = self.create_intersections()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def create_intersections(self):
        intersections = []
        for i in range(4):
            intersection = {"location": np.array([np.random.randint(0, self.size - 1),
                                                  np.random.randint(0, self.size - 1)]),
                            "colour": "green"}
            while (np.array_equal(self._agent_location, intersection["location"]) or
                   np.array_equal(self._target_location, intersection["location"])):
                intersection = {"location": np.array([np.random.randint(0, self.size - 1),
                                                      np.random.randint(0, self.size - 1)]),
                                "colour": "green"}
            intersections.append(intersection)

        return intersections

    def _get_obs(self):
        return [
            self.check_danger_direction("straight"),
            self.check_danger_direction("right"),
            self.check_danger_direction("left"),
            self.direction == 1,
            self.direction == 2,
            self.direction == 3,
            self.direction == 4,
            self._target_location[0] > self._agent_location[0],
            self._target_location[0] < self._agent_location[0],
            self._target_location[1] > self._agent_location[1],
            self._target_location[1] < self._agent_location[1]
        ]

    def _get_info(self):
        return np.linalg.norm(self._agent_location - self._target_location, ord=1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while self.roadblock_collision():
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        while self.roadblock_collision() or np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._obstacle_locations = [np.array([np.random.randint(1, self.size - 2),
                                              np.random.randint(1, self.size - 2)])
                                    for _ in range(10)]
        self._intersections = self.create_intersections()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # [straight, right, left, stop]
        cur_location = self._agent_location
        direction = dict(r=1, l=2, u=3, d=4)
        clockwise = [direction['r'], direction['d'], direction['l'], direction['u']]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0, 0]) or np.array_equal(action, [0, 0, 0, 1]):
            new_dir = clockwise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clockwise[next_idx]  # right turn r -> d -> l -> u
        else:
            next_idx = (idx - 1) % 4
            new_dir = clockwise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        if not np.array_equal(action, [0, 0, 0, 1]):
            if self.direction == direction["r"]:
                self._agent_location = np.clip(self._agent_location + np.array([1, 0]), 0, self.size - 1)
            elif self.direction == direction["l"]:
                self._agent_location = np.clip(self._agent_location + np.array([-1, 0]), 0, self.size - 1)
            elif self.direction == direction["u"]:
                self._agent_location = np.clip(self._agent_location + np.array([0, 1]), 0, self.size - 1)
            else:
                self._agent_location = np.clip(self._agent_location + np.array([0, -1]), 0, self.size - 1)

        game_over = False
        reward = 0
        if (self.roadblock_collision() or
                self.run_traffic_light() or
                not self.change_direction_intersection(action, cur_location)):
            game_over = True
            reward = -5
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, game_over, False, info

        if np.array_equal(self._agent_location, self._target_location):
            game_over = True
            reward = 10
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, game_over, False, info

        if np.random.randint(0, 10) > 6:
            self.place_obstacles()
            self.change_traffic_light_to_red()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, game_over, False, info

    def change_direction_intersection(self, action, location):
        if np.array_equal(action, [1, 0, 0, 0]) or np.array_equal(action, [0, 0, 0, 1]):
            return True
        intersection_locations = [intersection["location"] for intersection in self._intersections]
        if location in intersection_locations:
            return True
        return False

    def check_intersection_danger(self, location, intersection_locations):
        if location not in intersection_locations:
            return False
        for intersection in self._intersections:
            if np.array_equal(intersection["location"], location) and intersection["colour"] == "red":
                return True

    def check_danger_direction(self, direction):
        intersection_locations = [intersection["location"] for intersection in self._intersections]
        go_right = np.clip(self._agent_location + np.array([1, 0]), 0, self.size - 1)
        go_left = np.clip(self._agent_location + np.array([-1, 0]), 0, self.size - 1)
        go_up = np.clip(self._agent_location + np.array([0, 1]), 0, self.size - 1)
        go_down = np.clip(self._agent_location + np.array([0, -1]), 0, self.size - 1)

        if direction == "straight":
            return ((self.direction == 1 and
                     go_right in self._obstacle_locations and
                     self.check_intersection_danger(go_right, intersection_locations)) or
                    (self.direction == 2 and
                     go_left in self._obstacle_locations and
                     self.check_intersection_danger(go_left, intersection_locations)) or
                    (self.direction == 3 and
                     go_up in self._obstacle_locations and
                     self.check_intersection_danger(go_up, intersection_locations)) or
                    (self.direction == 4 and
                     go_down in self._obstacle_locations and
                     self.check_intersection_danger(go_down, intersection_locations)))

        if direction == "right":
            return ((self.direction == 1 and
                     go_down in self._obstacle_locations and
                     self.check_intersection_danger(go_down, intersection_locations)) or
                    (self.direction == 2 and
                     go_up in self._obstacle_locations and
                     self.check_intersection_danger(go_up, intersection_locations)) or
                    (self.direction == 3 and
                     go_right in self._obstacle_locations and
                     self.check_intersection_danger(go_right, intersection_locations)) or
                    (self.direction == 4 and
                     go_left in self._obstacle_locations and
                     self.check_intersection_danger(go_left, intersection_locations)))

        if direction == "left":
            return ((self.direction == 1 and
                     go_up in self._obstacle_locations and
                     self.check_intersection_danger(go_up, intersection_locations)) or
                    (self.direction == 2 and
                     go_down in self._obstacle_locations and
                     self.check_intersection_danger(go_down, intersection_locations)) or
                    (self.direction == 3 and
                     go_left in self._obstacle_locations and
                     self.check_intersection_danger(go_left, intersection_locations)) or
                    (self.direction == 4 and
                     go_right in self._obstacle_locations and
                     self.check_intersection_danger(go_right, intersection_locations)))

    def roadblock_collision(self):
        return self._agent_location in self._obstacle_locations

    def place_obstacles(self):
        self._obstacle_locations = []
        for i in range(10):
            obstacle_location = np.array([np.random.randint(0, self.size - 1),
                                          np.random.randint(0, self.size - 1)])
            while (np.array_equal(self._agent_location, obstacle_location) or
                   np.array_equal(self._target_location, obstacle_location)):
                obstacle_location = [np.random.randint(0, self.size - 1), np.random.randint(0, self.size - 1)]
            self._obstacle_locations.append(obstacle_location)

    def run_traffic_light(self):
        intersection_locations = [intersection["location"] for intersection in self._intersections]
        if self._agent_location not in intersection_locations:
            return False
        intersection_agent = None
        for intersection in intersection_locations:
            if np.array_equal(self._agent_location, intersection):
                intersection_agent = intersection
                break
        for intersection in self._intersections:
            if np.array_equal(intersection["location"], intersection_agent) and intersection["colour"] == "red":
                return True

    def change_traffic_light_to_red(self):
        self._intersections = [{**d, "colour": "red"} for d in self._intersections]
