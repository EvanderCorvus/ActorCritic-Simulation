import torch as tr
import torch.nn as nn
import numpy as np
import random
from collections import deque

#Deep Deterministic Policy Gradient
class Actor(nn.Module):
    def __init__(self,hidden_dim):#state-space: orientation, position, force
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
    #Action space: Absolute orientation
    def forward(self,state):
        return(2*np.pi*self.network(state))

class Critic(nn.Module):
    def __init__(self,hidden_dim):
        super(Critic,self).__init__()
        self.network = nn.Sequential(
            #state+action dim
            nn.Linear(6,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self,state,action):
        return self.network(tr.cat([state, action], dim=1))
    
# class Memory:
#     def __init__(self, max_size):
#         self.buffer = deque(maxlen=max_size)
    
#     def push(self, state, action, reward, next_state, done):
#         experience = (state, action, np.array([reward]), next_state, done)
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#         state_batch = []
#         action_batch = []
#         reward_batch = []
#         next_state_batch = []
#         done_batch = []

#         batch = random.sample(self.buffer, batch_size)

#         for experience in batch:
#             state, action, reward, next_state, done = experience
#             state_batch.append(state)
#             action_batch.append(action)
#             reward_batch.append(reward)
#             next_state_batch.append(next_state)
#             done_batch.append(done)
        
#         return state_batch, action_batch, reward_batch, next_state_batch, done_batch

#     def __len__(self):
#         return len(self.buffer)