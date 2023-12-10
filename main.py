import RoadNetEnv
import gymnasium as gym
import numpy as np
import util


def main():
    """
    Creates the graph, makes the environment, and runs a randomized or human-controlled.

    :return:
    """
    graph, locations = util.make_grid_city(5, 5)
    env = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations, single_target=True)
    observation, info = env.reset()

    print(observation)
    user_input = input("Manually move agent? (Y/N)\n")
    if user_input.lower() == "y":
        env.render()
        env.manual_input_mode()
    else:
        for _ in range(1000):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
