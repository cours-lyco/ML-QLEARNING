import numpy as np


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
