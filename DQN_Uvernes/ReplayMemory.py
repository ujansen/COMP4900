# Implementation followed the PyTorch DQN tutorial available at: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gymnasium as gym
import random
from collections import deque
from data_types import Transition


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)