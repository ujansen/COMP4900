import RoadNetEnv
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import a2c_sb3_training 
import math

def isNan(n):
    return n != n

def main():
    
    graph, locations = a2c_sb3_training.make_grid_city(5,5)
    # single targets
    # env = gym.make('RoadNetEnv-v0', render_mode=None, graph=graph, node_positions=locations, multi_target=False)
    # multiple targets
    env = gym.make('RoadNetEnv-v0', render_mode=None, graph=graph, node_positions=locations, multi_target=True)
    
    # model = A2C.load("a2c_RoadNetEnv_single_target", env=env)
    model = A2C.load("a2c_RoadNetEnv_multi_target", env=env)
    vec_env = model.get_env()

    # random environments
    # model = A2C("MultiInputPolicy", env, verbose=1)
    # model.learn(total_timesteps=0)
    # vec_env = model.get_env()

    # model = A2C.load("a2c_RoadNetEnv")
    episode_graph_rewards = []
    number_episodes = 100
    obs = vec_env.reset()
    episode_reward = []


    for episode in range(number_episodes):
        obs = vec_env.reset()
        terminated = False
        total_rewards = 0
        episode_rewards = []
        while not terminated: 
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = vec_env.step(action)
            terminated = done[0]
            if rewards != 0:
                # if testing multiple targets
                total_rewards+=rewards[0]
                episode_rewards.append(rewards[0])
                
                # if testing single targets
                # total_rewards+=rewards
                # episode_rewards.append(rewards)
            
    
        episode_reward.append(episode_rewards)
        episode_graph_rewards.append(total_rewards)
       
    model_reward = np.array([np.mean(np.array(x)) for x in episode_reward])
    
    for i in range(len(model_reward)):
        if isNan(model_reward[i]):
            model_reward[i] = 0
    
    print(model_reward, len(model_reward), sum(model_reward))
    
    plt.scatter(np.arange(len(episode_graph_rewards)), episode_graph_rewards, s=2)
    plt.title("Total reward per episode A2C (stablebaseline3)")
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.show()
    


    vec_env.close()



main()
