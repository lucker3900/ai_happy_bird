import pygame

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y, imgs):
        super().__init__()
        self.x = x
        self.y = y
        self.imgs = imgs
        self.img_index = 0
        self.image = self.imgs[self.img_index]
        self.down_speed = 0
        self.ground_height = 500  # 添加地面高度变量

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.frequency = 100
        self.last_update = pygame.time.get_ticks()

    def set_ground_height(self, height):
        # 设置地面高度
        self.ground_height = height

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.frequency:
            self.img_index += 1
            if self.img_index >= len(self.imgs):
                self.img_index = 0
            self.image = self.imgs[self.img_index]
            self.image = pygame.transform.rotate(self.image, -self.down_speed * 2)
            self.last_update = now

            self.down_speed += 3
            if self.down_speed > 9:
                self.down_speed = 9
            self.rect.y += self.down_speed

    def jump(self, click):
        if click:
            self.down_speed += -10
        else:
            self.rect.y += -5

    def drop(self):
        # 旋转小鸟朝下
        self.image = pygame.transform.rotate(self.imgs[self.img_index], -90)
        self.rect.bottom += 6
        if self.rect.bottom >= 500:
            self.rect.bottom = 500
