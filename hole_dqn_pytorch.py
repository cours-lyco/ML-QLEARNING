import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys
import copy
import argparse
import gym
import numpy as np
from itertools import count
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys
from tqdm import tqdm

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 0,  255)


WIDTH = 500
HEIGHT = 500
ROWS = 4

grid = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ]

R = np.array([
        [-1, -1, -1, -1],
        [-1, -5, -1, -5],
        [-1, -1, -1, -5],
        [-5, -1, -1, 10]
], dtype=np.int32)

actions = [
    (0, -1), #left
    (0, 1),  #right
    (-1, 0), #up
    (1, 0) #down
]

Q = np.zeros( (len(R[0,:])*len(R[:,0]) , len(actions)), dtype=np.float64 )


class Policy(nn.Module):
    def __init__(self, nbr_observation_space, nbr_action_space):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(nbr_observation_space, 32)
        self.Dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(32, nbr_action_space)

    def forward(self, x):
        x = self.affine1(x)
        x = self.Dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class Env(object):
    def __init__(self):

        self.state = 0
        self.start = [0,0]
        #self.state  = np.concatenate(self.state, axis=None)

        #self.state_copy = copy.deepcopy(self.state)
        #np.zeros((len(R[0,:]),len(R[:,0])), dtype=np.int32)


    def step(self, action):
        done = False

        self.state = np.zeros((len(R[0,:]),len(R[:,0])), dtype=np.int32)
        self.state = self.state.reshape((4,4))

        self.start[0] = max(0, min(self.start[0] + actions[action][0], -1 + len(R[:,0])  ))
        self.start[1] = max(0, min(self.start[1] + actions[action][1], -1 + len(R[:,0])  ))

        #self.state[self.start[0], self.start[1]] = 1
        #self.state = np.concatenate(self.state, axis=None)

        new_state = self.start[0] + 4*   self.start[1]
        if new_state == 15:
            done = True

        return new_state, R[self.start[0], self.start[1]], done

    def reset(self):
        self.start = [0,0]
        return 0


class Agent(object):
    def __init__(self, action_len):
        self.action_len = action_len

    def take_action(self, state):
        state_array = np.zeros((len(R[0,:]) * len(R[:,0])), dtype=np.int32)
        posx, posy = state//len(R[0,:]), state % len(R[:,0])

        state_array[ posx * 4 + posy ] = 1

        state = torch.from_numpy(state_array).float().unsqueeze(0)
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        action_index = action.item()
        return action_index


if __name__ == '__main__':
    lr = 0.1
    ep = 0.99579

    gamma = 0.88
    scores = []

    start, end = 0, 15

    nbr_observation_state, nbr_action_space = len(R[0,:])*len(R[:,0]), len(actions)

    policy = Policy(nbr_observation_state, nbr_action_space)

    env = Env()
    ag = Agent(4)

    num = 2000

    scores = []

    for i in tqdm(range(num)):
        done = False
        state = env.reset()
        while not done:
            action = ag.take_action(state)
            new_state, reward, done = env.step(action)
            Q[state, action] = Q[state, action] + lr*(reward + gamma*np.max(Q[new_state,:]) - Q[state, action])

            if np.max(Q) > 0:
                score = np.sum(Q/np.max(Q)*100)
            else:
                score = 0
            scores.append(score)
            state = new_state

    print(Q)
    state = env.reset()
    steps = [str(state)]
    done = False
    while not done:
        action =  np.argmax(Q[state,:])
        new_state, _, done = env.step(action)
        steps.append(str(new_state))
        state = new_state

print("-"*60)
print(" ".join(steps))
print("-"*60)
env.reset()

plt.plot(scores)
plt.show()
