# Implementation followed the PyTorch DQN tutorial available at: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 

from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
