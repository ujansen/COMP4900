import gymnasium as gym
from stable_baselines3 import PPO
import RoadNetEnv
from main import make_grid_city

# Assuming you've registered your environment with gym
graph, locations = make_grid_city(5, 5)
env_single = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations,
                      multi_target=False)
# env_multi = gym.make('RoadNetEnv-v0', render_mode='human', graph=graph, node_positions=locations, multi_target=True)

# Create the PPO model
model_single = PPO("MultiInputPolicy", env_single, verbose=1)
# model_multi = PPO("MultiInputPolicy", env_multi, verbose=1)

# Train the model
# model_single.learn(total_timesteps=10000)
model_single.learn(total_timesteps=10000)

model_single.save("ppo_model_different_reward_probability")
# model_single.save("ppo_model_different_reward")

