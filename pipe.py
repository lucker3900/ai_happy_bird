#管道向左移动,速度4
#每1500ms生成一个管道

import pygame

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, img, is_top):
        super().__init__()
        self.image = img
        self.rect = self.image.get_rect()
        self.speedx = 4
        self.frequency = 1500
        self.cross_pipe = True
        self.last_update = pygame.time.get_ticks()
        self.is_top = is_top

        if is_top:
            self.rect.bottomleft = (x, y)
        else:
            self.rect.topleft = (x, y)

    def update(self):
        self.rect.left -= self.speedx
        if self.rect.right < 0:
            self.kill()


