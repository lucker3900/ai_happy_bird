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

#分数记录
game_score = 0

class GenerationReporter(neat.reporting.BaseReporter):
    def __init__(self, game):
        self.game = game

    def start_generation(self, generation):
        self.game.generation = generation

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.font.init()  # 初始化字体模块
        
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NEAT Flappy Bird")
        
        # 初始化字体
        try:
            self.font = pygame.font.Font("微软正黑体.ttf", 30)
        except:
            # 如果找不到指定字体，使用系统默认字体
            self.font = pygame.font.Font(None, 30)
        
        self.pipe_distance = 150
        self.generation = 0
        self.birds = []
        self.nets = []
        self.ge = []
        self.score = 0  # 作为类属性
        
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
        # 绘制上下管道空隙中间的射线
        if len(pipes) >= 2:
            for i in range(0, len(pipes), 2):
                if i + 1 < len(pipes):
                    # 获取上下管道
                    top_pipe = pipes[i] if pipes[i].is_top else pipes[i+1]
                    bottom_pipe = pipes[i+1] if not pipes[i+1].is_top else pipes[i]
                    
                    # 计算空隙中间点
                    gap_center_x = top_pipe.rect.centerx
                    gap_center_y = (top_pipe.rect.bottom + bottom_pipe.rect.top) / 2
                    
                    # 绘制射线
                    pygame.draw.line(
                        self.window,
                        (0, 255, 0),  # 绿色
                        (gap_center_x - 20, gap_center_y),  # 左端点
                        (gap_center_x + 20, gap_center_y),  # 右端点
                        2  # 线宽
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
        
        # 绘制主射线（从小鸟到目标点）
        pygame.draw.line(
            self.window,
            (255, 0, 0),  # 红色
            bird_pos,
            target_pos,
            2  # 线宽
        )

    def eval_genomes(self, genomes, config):
        """评估每个基因组"""
        while True:
            self.birds = []
            self.nets = []
            self.ge = []
            self.pipe_sprite = pygame.sprite.Group()
            self.score = 0  # 重置分数
            
            # 计算小鸟之间的垂直间距
            total_birds = len(genomes)
            available_height = SCREEN_HEIGHT - 200  # 留出上下边距
            spacing = available_height / total_birds  # 动态计算间距
            
            # 初始化种群
            for i, (genome_id, genome) in enumerate(genomes):
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                self.nets.append(net)
                
                # 设置垂直排列的起始位置
                start_x = 100  # 固定X坐标
                start_y = 100 + i * spacing  # 使用动态间距
                
                bird = Bird(start_x, start_y, self.bird_imgs)
                bird.ground_height = SCREEN_HEIGHT - 100
                self.birds.append(bird)
                
                genome.fitness = 0
                self.ge.append(genome)

            clock = pygame.time.Clock()
            running = True
            last_pipe_time = pygame.time.get_ticks()

            while running and len(self.birds) > 0:
                clock.tick(FPS)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        raise pygame.error("User quit")

                # 创建新管道
                now = pygame.time.get_ticks()
                if now - last_pipe_time > PIPE_FREQUENCY:
                    pipe_top, pipe_btm = self.create_pipe()
                    self.pipe_sprite.add(pipe_top, pipe_btm)
                    last_pipe_time = now

                # 更新所有对象
                self.pipe_sprite.update()
                pipe_list = self.pipe_sprite.sprites()
                now = pygame.time.get_ticks()

                if len(pipe_list) > 0:
                    first_pipe = pipe_list[0]
                    if bird.rect.left > first_pipe.rect.right and first_pipe.cross_pipe:
                        print(f"第一根柱子坐标{first_pipe.rect.right}")
                        self.score += 1
                        first_pipe.cross_pipe = False

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

                # 显示分数和文字
                cc_text = self.font.render("CC无聊纯娱乐", True, (0, 0, 255))
                self.window.blit(cc_text, (SCREEN_WIDTH / 2 - 100, 10))
                score_text = self.font.render(f"score:{str(self.score)}", True, (0, 0, 255))
                self.window.blit(score_text, (SCREEN_WIDTH/2 - 50, 50))
                gen_text = self.font.render(f"gen:{self.generation}", True, (0, 0, 255))
                self.window.blit(gen_text, (SCREEN_WIDTH/2 - 50, 90))
                birds_text = self.font.render(f"alive:{len(self.birds)}", True, (0, 0, 255))
                self.window.blit(birds_text, (SCREEN_WIDTH/2 - 50, 130))

                pygame.display.update()

                # 当所有小鸟死亡时
                if len(self.birds) == 0:
                    self.generation += 1  # 增加代数
                    break  # 跳出内层循环，重新初始化种群

def run_neat(config_path):
    """运行NEAT算法"""
    try:
        print(f"正在读取配置文件: {config_path}")
        
        # 先读取原配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            
        # 创建临时配置文件（移除中文注释）
        temp_config_path = os.path.join(os.path.dirname(__file__), "temp_config.txt")
        with open(temp_config_path, 'w') as f:
            # 移除包含中文的注释行
            cleaned_content = '\n'.join(line for line in config_content.split('\n') 
                                      if not line.strip().startswith('#'))
            f.write(cleaned_content)
            
        # 使用临时配置文件
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           temp_config_path)
        
        # 删除临时配置文件
        os.remove(temp_config_path)
        
        print("配置文件读取成功，初始化种群...")
        p = neat.Population(config)
        
        # 添加统计报告
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        # 创建游戏实例
        game = Game()
        
        print("开始训练...")
        # 运行进化过程
        winner = p.run(game.eval_genomes)
        
        # 保存最佳网络
        with open("best.pickle", "wb") as f:
            pickle.dump(winner, f)
            
    except pygame.error:  # 捕获pygame退出异常
        print("训练被用户终止")
    except FileNotFoundError as e:
        print(f"找不到配置文件: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
        logging.error(f"Error in run_neat: {e}")
        raise

if __name__ == "__main__":
    # 获取配置文件的绝对路径
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    
    # 确保配置文件存在
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
        
    try:
        run_neat(config_path)  # 运行NEAT 
    except Exception as e:
        print(f"错误: {e}") 