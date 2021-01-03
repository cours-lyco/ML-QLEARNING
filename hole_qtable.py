import numpy as np
from random import randint
import random
import pylab as plt
import sys
from math import pow
import time

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

class Env(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.x = self.start % len(R[:,0])
        self.y = self.end // len(R[0,:])
        self.tile_size = WIDTH//ROWS
        self.centers = [ ]
        for i in range(ROWS):
            x = self.tile_size*i
            for j in range(ROWS):
                y = self.tile_size*j
                self.centers.append( (j*ROWS+i,   x + self.tile_size//2 , y + self.tile_size//2 ))

    def step(self, action):
        done = False
        self.x = max(0, min(self.x + actions[action][0], -1 + len(R[:,0])  ))
        self.y = max(0, min(self.y + actions[action][1], -1 + len(R[:,0])  ))

        self.start = self.y * len(R[0,:])  + self.x
        if self.start == self.end:
            done = True
        return self.start, R[self.y, self.x], done

    def reset(self):
        self.x = 0
        self.y = 0
        self.start = 0
        return 0

    def render(self,step):
        import pygame

        def draw_agent_current_state( step, win, image_agent):
            state, xcenter, ycenter = list(filter(lambda x: x[0] == int(step), self.centers))[0]

            win.blit(image_agent, ((2*xcenter - image_agent.get_width())//2, (2*ycenter - image_agent.get_height()) //2)   )
            #time.sleep(2500)

        def draw_env(win, image_agent, step):
            #draw grid
            for i in range(ROWS):
                x = i*self.tile_size
                #vertical
                pygame.draw.line(win, WHITE, (x, 0), (x, HEIGHT), 3)
                #horizontal
                pygame.draw.line(win, WHITE, (0, x), (WIDTH, x), 3)

                draw_agent_current_state(step, win, image_agent )
                color = BLACK
                for j in range(ROWS):
                    y = j*self.tile_size
                    if grid[j][i] == 'S':
                        color = GREEN
                    if grid[j][i] == 'F':
                        color = BLACK
                    elif grid[j][i] == 'G':
                        color = ORANGE
                    elif grid[j][i] == 'H':
                        color = RED
                    pygame.draw.rect(win,  color, (x+2, y+2,self.tile_size-2, self.tile_size-2))

            pygame.display.flip()



        pygame.init()
        win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('MathsPhysic Code')
        size=80
        image_agent = pygame.transform.scale(pygame.image.load("img/agent_reinforcement.png"), (size, size))

        done = False
        i = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    #pygame.quit()
                    done = True
                    sys.exit(2)

            if i >= len(steps):
                i = 0
            draw_env(win, image_agent, steps[i])
            pygame.time.delay(500)
            i += 1
            #pygame.display_flip()


class Agent(object):
    def __init__(self, action_len):
        self.action_len = action_len

    def take_action(self, state, eps):
        if np.random.normal(0, 1) < eps:
            action = np.random.randint(0, self.action_len)
        else:
            action = np.argmax(state)
        return action

if __name__ == '__main__':
    lr = 0.1
    ep = 0.99579
    epsilon = 0.4
    num_episod = 1000
    gamma = 0.88
    scores = []

    start, end = 0, 15

    env = Env(start, end)
    ag = Agent(len(actions))

    for episode in range(num_episod):
        done = False
        state = env.reset()
        epsilon = pow(ep, episode)
        while not done :
            action = ag.take_action(Q[state,:], epsilon)
            new_state, reward, done = env.step(action)

            Q[state, action] = Q[state, action] + lr*(reward + gamma*np.max(Q[new_state,:]) - Q[state, action])

            if np.max(Q) > 0:
                score = np.sum(Q/np.max(Q)*100)
            else:
                score = 0
            scores.append(score)
            state = new_state

    state = env.reset()
    steps = [str(state)]
    while state != 15:
        action =  np.argmax(Q[state,:])
        new_state, reward, done = env.step(action)
        steps.append(str(new_state))

        state = new_state
env.reset()
print("-"*60)
print(" ".join(steps))
print("-"*60)

env.render(steps)


#plt.plot(scores)
#plt.show()
