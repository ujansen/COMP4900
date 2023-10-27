import util
import RoadNetEnv
import gymnasium as gym
import numpy as np

def main():
    graph = util.make_grid_city(10, 10)

    env = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph)

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
