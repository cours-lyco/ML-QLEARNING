import pygame
import gym
#from gym import envs
from gym import spaces
import numpy as np
#print(envs.registry.all())
from stable_baselines.common.env_checker import check_env

#https://towardsdatascience.com/ultimate-guide-for-reinforced-learning-part-1-creating-a-game-956f1f2b0a91

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.width = 500
        self.height = 200

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def step(self, action):
            return new_state, reward, done , info

    def reset(self):
        return self.observation

    def render(self):
        import pygame
        pygame.init()
        window = pygame.set_mode(self.width, self.height)
        done = False
        while not done:
            for event in pygame.get_event():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        pass
                if event.type == pygame.QUIT:
                    done = True

if __name__ == '__main__':
    nbr_action, num_row, num_col = 4, 3,3
    env = CustomEnv(nbr_action, num_row, num_col)

    print(env.action_space.sample())
