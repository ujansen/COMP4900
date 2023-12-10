import gymnasium as gym
from stable_baselines3 import PPO
import RoadNetEnv
from main import make_grid_city
import time
import matplotlib.pyplot as plt
import numpy as np

graph, locations = make_grid_city(5, 5)
env_single = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations, multi_target=False)
# env_multi = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations, multi_target=True)
model_single = PPO.load("ppo_model_different_reward_probability", env=env_single)
# model_multi = PPO.load("ppo_model_tsp_same_reward", env=env_multi)


vec_env = model_single.get_env()
vec_env_random = env_single
obs_random, _ = vec_env_random.reset()
model_reward = []
random_reward = []

for i in range(100):
    obs = vec_env.reset()
    model_rewards = []
    while True:
        action, _states = model_single.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)
        if rewards != 0:
            model_rewards.append(rewards)
        if done:
            break
    model_reward.append(model_rewards)

model_reward = np.array([np.mean(np.array(x)) for x in model_reward])
print(model_reward, len(model_reward), sum(model_reward))


# for i in range(100):
#     random_rewards = []
#     while True:
#         _, reward, done, _, _ = vec_env_random.step(vec_env_random.action_space.sample())
#         # # time.sleep(1)
#         if reward != 0:
#             random_rewards.append(reward)
#         if done:
#             _, _ = vec_env_random.reset()
#             break
#     random_reward.append(random_rewards)
#
# random_reward = np.array([np.mean(np.array(x)) for x in random_reward])
# print(random_reward, len(random_reward), sum(random_reward))
