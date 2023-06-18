import torch as tr
import torch.nn as nn
import numpy as np
import random
from collections import deque

#Deep Deterministic Policy Gradient
class Actor(nn.Module):
    def __init__(self,hidden_dim):#state-space: orientation, position, force
        super(Actor, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(5,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2)
        )
    #Action space: Absolute orientation
    def forward(self,state):
        musigma = self.actor_network(state)
        
        mu, sigma = musigma[:,0], musigma[:,1]
        return mu, sigma

class Critic(nn.Module):
    def __init__(self,hidden_dim):
        super(Critic,self).__init__()
        self.critic_network = nn.Sequential(
            #state+action dim
            nn.Linear(6,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self,state,action):

        state_action = tr.cat([state, action[:,None]], dim=1)
        return self.critic_network(state_action)
    
class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)