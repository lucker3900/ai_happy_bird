import pygame
import neat
import numpy as np
import random
import os
import sys
import math
import logging
from typing import List, Tuple
import pickle
from bird import Bird  # 导入已有的Bird类
from pipe import Pipe  # 导入已有的Pipe类

# 设置日志
logging.basicConfig(
    filename='ai_game_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 游戏窗口设置
SCREEN_WIDTH = 780
SCREEN_HEIGHT = 600
FPS = 60
PIPE_FREQUENCY = 1500  # 管道生成频率

class GenerationReporter(neat.reporting.BaseReporter):
    def __init__(self, game):
        self.game = game

    def start_generation(self, generation):
        self.game.generation = generation

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NEAT Flappy Bird")
        
        self.pipe_distance = 150
        self.generation = 0
        self.birds = []
        self.nets = []
        self.ge = []
        
        self.load_resources()
        
    def load_resources(self):
        try:
            self.bg_img = pygame.image.load("img/bg.png")
            self.bg_img = pygame.transform.scale(self.bg_img, (780, 600))
            self.ground_img = pygame.image.load("img/ground.png")
            self.pipe_btm_img = pygame.image.load("img/pipe.png")
            self.pipe_top_img = pygame.transform.flip(self.pipe_btm_img, False, True)
            self.bird_imgs = [
                pygame.image.load(f"img/bird{i}.png") for i in range(1, 3)
            ]
        except Exception as e:
            logging.error(f"Error loading resources: {e}")
            raise
            
    def create_pipe(self):
        """创建新的管道"""
        pipe_y = random.randint(-100, 100)
        # 确保正确传递is_top参数
        pipe_btm = Pipe(
            x=SCREEN_WIDTH, 
            y=SCREEN_HEIGHT/2 + self.pipe_distance/2 + pipe_y,
            img=self.pipe_btm_img, 
            is_top=False  # 明确指定参数名
        )
        pipe_top = Pipe(
            x=SCREEN_WIDTH, 
            y=SCREEN_HEIGHT/2 - self.pipe_distance/2 + pipe_y,
            img=self.pipe_top_img, 
            is_top=True   # 明确指定参数名
        )
        return pipe_top, pipe_btm

    def get_pipe_center(self, pipes):
        """获取最近的管道间隙中心点"""
        for pipe in pipes:
            if pipe.rect.right > 100:  # 小鸟的x位置
                if not pipe.is_top:  # 找到下管道，使用is_top而不是top
                    return (pipe.rect.centerx, pipe.rect.top - self.pipe_distance/2)
        return (SCREEN_WIDTH, SCREEN_HEIGHT/2)

    def draw_sensors(self, bird, pipes):
        """绘制从小鸟到管道中心的视觉传感器"""
        if not pipes:
            return
            
        bird_pos = np.array([bird.rect.centerx, bird.rect.centery])
        target_pos = np.array(self.get_pipe_center(pipes))
        
        # 绘制主射线（指向管道中心）
        pygame.draw.line(
            self.window,
            (255, 0, 0),  # 红色
            bird_pos,
            target_pos,
            2  # 线宽
        )
        
        # 绘制扇形射线
        angles = np.linspace(-30, 30, 10)  # 10条射线，覆盖60度角
        for angle in angles:
            rad = math.radians(angle)
            direction = target_pos - bird_pos
            length = np.linalg.norm(direction)
            
            # 旋转向量
            rotated = np.array([
                direction[0] * math.cos(rad) - direction[1] * math.sin(rad),
                direction[0] * math.sin(rad) + direction[1] * math.cos(rad)
            ])
            
            # 归一化并设置长度
            rotated = rotated / np.linalg.norm(rotated) * length
            end_pos = bird_pos + rotated
            
            pygame.draw.line(
                self.window,
                (255, 0, 0),
                bird_pos,
                end_pos,
                1
            )

    def draw_debug_info(self, bird, pipes):
        """绘制调试信息，包括物体边框和视觉传感器"""
        # 1. 绘制小鸟的边框
        pygame.draw.rect(self.window, (255, 0, 0), bird.rect, 2)
        
        # 2. 绘制管道的边框
        for pipe in pipes:
            pygame.draw.rect(self.window, (255, 0, 0), pipe.rect, 2)
        
        # 3. 绘制视觉传感器射线
        if not pipes:
            return
        
        bird_pos = np.array([bird.rect.centerx, bird.rect.centery])
        
        # 获取最近的管道间隙中心点
        pipe_center = self.get_pipe_center(pipes)
        target_pos = np.array(pipe_center)
        
        # 计算射线角度范围
        num_rays = 20  # 射线数量
        angle_range = 60  # 总角度范围
        angles = np.linspace(-angle_range/2, angle_range/2, num_rays)
        
        # 绘制每条射线
        for angle in angles:
            rad = math.radians(angle)
            direction = target_pos - bird_pos
            length = np.linalg.norm(direction)
            
            # 旋转向量
            rotated = np.array([
                direction[0] * math.cos(rad) - direction[1] * math.sin(rad),
                direction[0] * math.sin(rad) + direction[1] * math.cos(rad)
            ])
            
            # 归一化并设置长度
            rotated = rotated / np.linalg.norm(rotated) * length
            end_pos = bird_pos + rotated
            
            # 绘制射线
            pygame.draw.line(
                self.window,
                (255, 0, 0),  # 红色
                bird_pos,
                end_pos,
                1  # 线宽
            )

    def eval_genomes(self, genomes, config):
        """评估每个基因组"""
        self.birds = []
        self.nets = []
        self.ge = []
        self.pipe_sprite = pygame.sprite.Group()
        
        # 初始化种群
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.nets.append(net)
            
            bird = Bird(100, SCREEN_HEIGHT/2, self.bird_imgs)
            bird.ground_height = SCREEN_HEIGHT - 100
            self.birds.append(bird)
            
            genome.fitness = 0
            self.ge.append(genome)

        clock = pygame.time.Clock()
        running = True
        score = 0
        last_pipe_time = pygame.time.get_ticks()

        while running and len(self.birds) > 0:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 创建新管道
            now = pygame.time.get_ticks()
            if now - last_pipe_time > PIPE_FREQUENCY:
                pipe_top, pipe_btm = self.create_pipe()
                self.pipe_sprite.add(pipe_top, pipe_btm)
                last_pipe_time = now

            # 更新所有对象
            self.pipe_sprite.update()
            pipe_list = self.pipe_sprite.sprites()

            # 对每个存活的小鸟进行神经网络控制
            for x, bird in enumerate(self.birds):
                # 增加存活奖励
                self.ge[x].fitness += 0.1
                
                # 获取到最近管道中心的距离和角度
                pipe_center = self.get_pipe_center(pipe_list)
                bird_pos = np.array([bird.rect.centerx, bird.rect.centery])
                target_pos = np.array(pipe_center)
                
                # 计算输入特征
                dx = target_pos[0] - bird_pos[0]
                dy = target_pos[1] - bird_pos[1]
                velocity = bird.down_speed
                
                # 神经网络决策
                output = self.nets[x].activate((dy, dx, velocity))
                if output[0] > 0.5:
                    bird.jump(True)

                bird.update()

                # 碰撞检测
                if pygame.sprite.spritecollideany(bird, self.pipe_sprite) or \
                   bird.rect.top <= 0 or bird.rect.bottom >= SCREEN_HEIGHT - 100:
                    self.ge[x].fitness -= 1
                    self.birds.pop(x)
                    self.nets.pop(x)
                    self.ge.pop(x)

            # 绘制游戏画面
            self.window.blit(self.bg_img, (0, 0))
            self.pipe_sprite.draw(self.window)
            
            for bird in self.birds:
                self.draw_debug_info(bird, pipe_list)
                self.window.blit(bird.image, bird.rect)
                
            self.window.blit(self.ground_img, (0, SCREEN_HEIGHT - 100))

            # 在绘制信息之前更新generation
            if len(self.birds) == 0:
                self.generation += 1  # 当所有小鸟死亡时增加代数

            # 显示信息
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Birds Alive: {len(self.birds)}", True, (255, 255, 255))
            gen_text = font.render(f"Generation: {self.generation}", True, (255, 255, 255))
            
            self.window.blit(score_text, (10, 10))
            self.window.blit(gen_text, (10, 50))
            
            pygame.display.update()

def run_neat(config_path):
    """运行NEAT算法"""
    try:
        # 使用UTF-8编码读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               config_path)
        
        p = neat.Population(config)
        
        # 添加统计报告
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(GenerationReporter(Game()))  # 添加generation reporter
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        # 创建游戏实例
        game = Game()
        
        # 运行进化过程
        winner = p.run(game.eval_genomes, 50)  # 50代
        
        # 保存最佳网络
        with open("best.pickle", "wb") as f:
            pickle.dump(winner, f)
            
    except Exception as e:
        logging.error(f"Error in run_neat: {e}")
        raise

def create_config():
    """创建NEAT配置文件"""
    config_content = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 100
pop_size             = 50
reset_on_extinction  = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate = 0.0
activation_options     = tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob       = 0.2

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate    = 0.01

# connection weight options
weight_init_mean       = 0.0
weight_init_stdev      = 1.0
weight_max_value       = 30
weight_min_value       = -30
weight_mutate_power    = 0.5
weight_mutate_rate     = 0.8
weight_replace_rate    = 0.1

# network parameters
num_hidden             = 0
num_inputs             = 3
num_outputs            = 1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

feed_forward            = True
initial_connection      = full

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation      = 20
species_elitism     = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2"""

    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    return config_path

if __name__ == "__main__":
    config_path = create_config()  # 创建配置文件
    run_neat(config_path)  # 运行NEAT 