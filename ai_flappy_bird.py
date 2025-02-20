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
import time
import tkinter as tk
from tkinter import ttk
import threading

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

class StatsWindow:
    def __init__(self):
        # 使用更小的字体
        self.font = pygame.font.Font('freesansbold.ttf', 16)
        self.stats_text = ""
        self.history_text = []  # 存储历史记录
        
    def draw(self, window, game_surface):
        """绘制统计信息"""
        # 在主窗口上绘制游戏画面和统计信息
        window.fill((50, 50, 50))  # 深灰色背景
        window.blit(game_surface, (0, 0))
        
        x = SCREEN_WIDTH + 20
        y = 10
        
        # 先绘制当前代的信息
        if self.stats_text.strip():
            for line in self.stats_text.split('\n'):
                if line.strip():
                    text_surface = self.font.render(line, True, (255, 255, 255))
                    window.blit(text_surface, (x, y))
                    y += 20
                else:
                    y += 10
            
            # 在当前代和历史记录之间添加蓝色分隔线
            if self.history_text:
                y += 5  # 给分隔线留出空间
                pygame.draw.line(window, (0, 150, 255), 
                               (SCREEN_WIDTH + 10, y), 
                               (SCREEN_WIDTH + 380, y), 2)
                y += 15  # 分隔线后的间距
        
        # 绘制历史记录
        for i, old_text in enumerate(self.history_text):
            for line in old_text.split('\n'):
                if line.strip():
                    text_surface = self.font.render(line, True, (200, 200, 200))
                    window.blit(text_surface, (x, y))
                    y += 20
                else:
                    y += 10
            
            # 在每条历史记录之间添加蓝色分隔线（除了最后一条）
            if i < len(self.history_text) - 1:
                y += 5
                pygame.draw.line(window, (0, 150, 255), 
                               (SCREEN_WIDTH + 10, y), 
                               (SCREEN_WIDTH + 380, y), 2)
                y += 15

    def update_stats(self, stats_text, is_generation_end=False):
        """更新统计信息"""
        if is_generation_end:
            # 将当前信息添加到历史记录的开头
            self.history_text.insert(0, self.stats_text)
            # 只保留最近几代的历史
            self.history_text = self.history_text[:3]
        self.stats_text = stats_text

class DetailedReporter(neat.reporting.BaseReporter):
    """自定义报告器，用于详细显示训练过程"""
    def __init__(self, game):
        self.game = game
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        """当新的一代开始时调用"""
        self.generation = generation
        self.generation_start_time = time.time()
        
        # 构建初始显示文本，去掉多余的空行
        stats_text = f"""******* Running generation {generation} *******
Population's average fitness: 0.00000 
Best fitness: 0.00000
Population size: {len(self.game.birds)}
Alive: {len(self.game.birds)}
Score: 0
ID    age  size  fitness  adj fit  stag
====  ===  ====  =======  =======  ===="""
        self.update_display(stats_text)

    def end_generation(self, config, population, species_set):
        """当一代结束时调用"""
        ng = len(population)
        ns = len(species_set.species)
        
        # 计算适应度统计
        fit_mean = np.mean([c.fitness for c in population.values()])
        fit_std = np.std([c.fitness for c in population.values()])
        fit_max = np.max([c.fitness for c in population.values()])
        
        # 构建显示文本
        stats_text = f"""
******* Generation {self.generation} Summary *******

Population's average fitness: {fit_mean:.5f} 
Standard deviation: {fit_std:.5f}
Best fitness: {fit_max:.5f}

Population size: {ng}
Species count: {ns}

ID    age  size  fitness  adj fit  stag
====  ===  ====  =======  =======  ===="""
        
        # 添加物种信息
        for sid, s in species_set.species.items():
            a = self.generation - s.created
            n = len(s.members)
            f = "--" if s.fitness is None else f"{s.fitness:.1f}"
            af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
            st = self.generation - s.last_improved
            stats_text += f"\n{sid:<4}  {a:>3}  {n:>4}  {f:>7}  {af:>7}  {st:>4}"
        
        # 添加时间信息
        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        stats_text += f"\nGeneration time: {elapsed:.3f} sec"
        
        if len(self.generation_times) > 0:
            mean_time = np.mean(self.generation_times)
            stats_text += f"\nMean generation time: {mean_time:.3f} sec"
        
        stats_text += f"\nTotal extinctions: {self.num_extinctions}"
        
        # 更新显示时标记这是代的结束
        self.update_display(stats_text, is_generation_end=True)

    def species_stagnant(self, sid, species):
        """当物种停滞时调用"""
        self.num_extinctions += 1

    def update_display(self, stats_text, is_generation_end=False):
        """更新显示窗口"""
        self.game.stats_window.update_stats(stats_text, is_generation_end)

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.font.init()  # 初始化字体模块
        
        # 创建一个更宽的窗口来容纳游戏和统计信息
        self.window = pygame.display.set_mode((SCREEN_WIDTH + 400, SCREEN_HEIGHT))
        pygame.display.set_caption("NEAT Flappy Bird")
        
        # 创建主游戏surface
        self.game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # 初始化统计窗口
        self.stats_window = StatsWindow()
        
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
            self.game_surface,
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
                    gap_center_y = (top_pipe.rect.bottom + bottom_pipe.rect.top) / 2 + 70
                    
                    # 绘制射线  
                    pygame.draw.line(
                        self.game_surface,
                        (0, 255, 0),  # 绿色
                        (gap_center_x - 20, gap_center_y),  # 左端点
                        (gap_center_x + 20, gap_center_y),  # 右端点
                        2  # 线宽
                    )

    def draw_debug_info(self, bird, pipes):
        """绘制调试信息，包括物体边框和视觉传感器"""
        # 1. 绘制小鸟的边框
        pygame.draw.rect(self.game_surface, (255, 0, 0), bird.rect, 2)
        
        # 2. 绘制管道的边框
        for pipe in pipes:
            pygame.draw.rect(self.game_surface, (255, 0, 0), pipe.rect, 2)
        
        # 3. 绘制视觉传感器射线
        if not pipes:
            return
        
        bird_pos = np.array([bird.rect.centerx, bird.rect.centery])
        
        # 获取最近的管道间隙中心点
        pipe_center = self.get_pipe_center(pipes)
        target_pos = np.array(pipe_center)
        
        # 绘制主射线（从小鸟到目标点）
        pygame.draw.line(
            self.game_surface,
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

                # 在game_surface上绘制游戏画面
                self.game_surface.blit(self.bg_img, (0, 0))
                self.pipe_sprite.draw(self.game_surface)
                
                for bird in self.birds:
                    self.draw_debug_info(bird, pipe_list)
                    self.game_surface.blit(bird.image, bird.rect)
                    
                self.game_surface.blit(self.ground_img, (0, SCREEN_HEIGHT - 100))

                # 显示分数和文字
                cc_text = self.font.render("CC无聊纯娱乐", True, (0, 0, 255))
                self.game_surface.blit(cc_text, (SCREEN_WIDTH / 2 - 100, 10))
                score_text = self.font.render(f"score:{str(self.score)}", True, (0, 0, 255))
                self.game_surface.blit(score_text, (SCREEN_WIDTH/2 - 50, 50))

                # 计算当前状态
                current_fitness = np.mean([g.fitness for g in self.ge]) if self.ge else 0
                max_fitness = max([g.fitness for g in self.ge]) if self.ge else 0
                
                # 更新统计信息时调整格式
                stats_text = f"""
******* Running generation {self.generation} *******

Population's average fitness: {current_fitness:.5f}
Best fitness: {max_fitness:.5f}

Population size: {len(self.birds)}
Alive: {len(self.birds)}
Score: {self.score}

ID    age  size  fitness  adj fit  stag
====  ===  ====  =======  =======  ===="""

                # 添加对齐的数据行
                for i, (genome_id, genome) in enumerate(zip(range(len(self.ge)), self.ge)):
                    if i < 5:  # 只显示前5个物种的信息
                        stats_text += f"\n{i:<4}  {0:>3}  {3:>4}  {genome.fitness:>7.1f}  {'-':>7}  {0:>4}"

                # 更新显示（不再添加分隔行和下一代标题）
                self.stats_window.update_stats(stats_text)
                self.stats_window.draw(self.window, self.game_surface)
                pygame.display.flip()

                # 当所有小鸟死亡时
                if len(self.birds) == 0:
                    self.generation += 1  # 增加代数
                    break  # 跳出内层循环，重新初始化种群

def run_neat(config_path):
    """运行NEAT算法"""
    try:
        print(f"正在读取配置文件: {config_path}")
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        p = neat.Population(config)
        
        # 创建游戏实例
        game = Game()
        
        # 添加详细报告器（传入game实例）
        detailed_reporter = DetailedReporter(game)
        p.add_reporter(detailed_reporter)
        
        # 运行进化过程
        winner = p.run(game.eval_genomes)
        
        # 保存最佳网络
        with open("best.pickle", "wb") as f:
            pickle.dump(winner, f)
            
    except pygame.error:
        print("训练被用户终止")
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