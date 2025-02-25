import pygame
import neat
import numpy as np
import random
import os
import sys
import logging
from typing import List, Tuple
import pickle
from bird import Bird  # 导入已有的Bird类
from pipe import Pipe  # 导入已有的Pipe类
import time
import tkinter as tk
from tkinter import ttk
import torch
import glob
import json

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

# 添加性能相关的常量
DRAW_DEBUG = False  # 关闭调试信息显示
DEBUG_UPDATE_INTERVAL = 3  # 调试信息更新间隔（帧数）
STATS_UPDATE_INTERVAL = 2  # 统计信息更新间隔（帧数）
MAX_PIPE_PAIRS = 3  # 最大管道对数
POPULATION_SIZE = 30  # 种群大小

# 添加新的常量
MIN_SCORE_TO_SAVE = 100  # 只保存达到这个分数的游戏数据
MAX_TRAINING_FILES = 10  # 最多保存10个高分记录
BACKUP_DIR = "training_data_backup"  # 备份目录

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
        pygame.display.set_caption("CC Happy Bird")
        
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
        self.best_score = 0  # 当前运行的最佳分数
        self.all_time_best_score = 0  # 历史最佳分数
        self.scores_history = []  # 记录所有分数
        self.checkpoint_interval = 10  # 每10代保存一次
        self.model_dir = "checkpoints"  # 保存模型的目录
        
        # 创建保存模型的目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.load_resources()
        
        # 尝试加载历史最佳分数
        self.load_best_score()
        
        self.frame_count = 0
        self.last_pipe_center = None
        self.last_pipe = None
        
        # 创建sprite组用于优化碰撞检测
        self.bird_group = pygame.sprite.Group()
        
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
        """移除调试信息绘制"""
        pass  # 不绘制任何调试信息

    def load_best_score(self):
        """加载历史最佳分数"""
        try:
            if os.path.exists('best_score.txt'):
                with open('best_score.txt', 'r') as f:
                    self.all_time_best_score = int(f.read())
                print(f"Loaded all-time best score: {self.all_time_best_score}")
        except Exception as e:
            print(f"Error loading best score: {e}")
            
    def save_best_score(self):
        """保存历史最佳分数"""
        try:
            with open('best_score.txt', 'w') as f:
                f.write(str(self.all_time_best_score))
        except Exception as e:
            print(f"Error saving best score: {e}")
            
    def save_checkpoint(self, genomes, generation):
        """保存检查点和分数记录"""
        try:
            # 找到当前代中适应度最高的基因组
            best_genome = None
            best_fitness = float('-inf')
            
            for _, genome in genomes:
                if genome.fitness is not None and genome.fitness > best_fitness:
                    best_genome = genome
                    best_fitness = genome.fitness
            
            if best_genome is None:
                return  # 如果没有有效的基因组，直接返回
            
            # 更新最高分记录
            if self.score > self.all_time_best_score:
                self.all_time_best_score = self.score
                self.save_best_score()
                print(f"New all-time best score: {self.all_time_best_score}!")
            
            # 记录本次分数
            self.scores_history.append({
                'generation': generation,
                'score': self.score,
                'best_fitness': best_fitness
            })
            
            # 保存分数历史
            with open('scores_history.json', 'w') as f:
                json.dump(self.scores_history, f, indent=2)
            
            checkpoint = {
                'generation': generation,
                'best_genome': best_genome,
                'best_fitness': best_fitness,
                'score': self.score,
                'all_time_best_score': self.all_time_best_score
            }
            
            # 保存常规检查点
            if generation % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.model_dir, 
                    f'checkpoint_gen_{generation}.pth'
                )
                torch.save(checkpoint, checkpoint_path)
            
            # 如果是最佳分数，单独保存
            if self.score > self.best_score:
                self.best_score = self.score
                best_model_path = os.path.join(
                    self.model_dir, 
                    f'best_model_score_{self.score}.pth'
                )
                torch.save(checkpoint, best_model_path)
                print(f"New best score! Saved model with score {self.score}")
                
        except Exception as e:
            print(f"Error in save_checkpoint: {e}")

    def eval_genomes(self, genomes, config):
        """优化的基因组评估"""
        self.frame_count = 0
        
        while True:
            # 重置游戏状态
            self.birds = []
            self.nets = []
            self.ge = []
            self.pipe_sprite = pygame.sprite.Group()
            self.bird_group = pygame.sprite.Group()  # 使用新的group
            self.score = 0
            
            # 初始化种群
            for i, (genome_id, genome) in enumerate(genomes):
                if i >= POPULATION_SIZE:  # 限制种群大小
                    break
                    
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                self.nets.append(net)
                
                bird = Bird(100, 100 + i * 20, self.bird_imgs)
                bird.ground_height = SCREEN_HEIGHT - 100
                self.birds.append(bird)
                self.bird_group.add(bird)
                
                genome.fitness = 0
                self.ge.append(genome)

            # 如果没有有效的小鸟，直接进入下一代
            if not self.birds:
                self.generation += 1
                break

            clock = pygame.time.Clock()
            running = True
            last_pipe_time = pygame.time.get_ticks()

            while running and self.birds:
                self.frame_count += 1
                clock.tick(FPS)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        raise pygame.error("User quit")

                # 创建新管道（带数量限制）
                now = pygame.time.get_ticks()
                if (now - last_pipe_time > PIPE_FREQUENCY and 
                    len(self.pipe_sprite) < MAX_PIPE_PAIRS * 2):
                    pipe_top, pipe_btm = self.create_pipe()
                    self.pipe_sprite.add(pipe_top, pipe_btm)
                    last_pipe_time = now

                # 更新和清理管道
                for pipe in list(self.pipe_sprite):
                    if pipe.rect.right < 0:
                        pipe.kill()
                self.pipe_sprite.update()

                # 优化的碰撞检测
                collisions = pygame.sprite.groupcollide(
                    self.bird_group, 
                    self.pipe_sprite, 
                    False, 
                    False
                )
                
                # 更新小鸟和神经网络
                pipe_list = self.pipe_sprite.sprites()
                birds_to_remove = []
                
                # 先更新所有小鸟
                for x, bird in enumerate(self.birds):
                    # 检查碰撞
                    if (bird in collisions or 
                        bird.rect.top <= 0 or 
                        bird.rect.bottom >= SCREEN_HEIGHT - 100):
                        birds_to_remove.append(x)
                        continue
                        
                    # 增加存活奖励
                    self.ge[x].fitness += 0.1
                    
                    # 获取神经网络输入
                    if pipe_list:
                        pipe_center = self.get_pipe_center(pipe_list)
                        bird_pos = np.array([bird.rect.centerx, bird.rect.centery])
                        dx = pipe_center[0] - bird_pos[0]
                        dy = pipe_center[1] - bird_pos[1]
                        
                        # 神经网络决策
                        output = self.nets[x].activate((dy, dx, bird.down_speed))
                        if output[0] > 0.5:
                            bird.jump(True)
                    
                    bird.update()

                # 从后向前移除死亡的小鸟
                for idx in sorted(birds_to_remove, reverse=True):
                    try:
                        if idx < len(self.birds):
                            bird = self.birds[idx]
                            self.ge[idx].fitness -= 1
                            self.birds.pop(idx)
                            self.nets.pop(idx)
                            self.ge.pop(idx)
                            self.bird_group.remove(bird)  # 直接使用bird对象而不是索引
                    except Exception as e:
                        print(f"Error removing bird at index {idx}: {e}")
                        continue

                # 检查是否所有小鸟都死亡
                if not self.birds:
                    self.generation += 1
                    break

                # 绘制游戏画面
                self.game_surface.blit(self.bg_img, (0, 0))
                self.pipe_sprite.draw(self.game_surface)
                
                for bird in self.birds:
                    self.game_surface.blit(bird.image, bird.rect)
                
                self.game_surface.blit(self.ground_img, (0, SCREEN_HEIGHT - 100))

                # 更新统计信息（降低频率）
                if self.frame_count % STATS_UPDATE_INTERVAL == 0:
                    self.update_stats(genomes)
                
                pygame.display.flip()

                # 检查是否需要保存检查点
                if self.frame_count % (FPS * 30) == 0:  # 每30秒保存一次
                    self.save_checkpoint(genomes, self.generation)

                if not self.birds:
                    self.generation += 1
                    break

                # 更新管道和检查得分
                pipe_list = self.pipe_sprite.sprites()
                if pipe_list:
                    # 检查每对管道
                    for i in range(0, len(pipe_list), 2):
                        if i + 1 < len(pipe_list):
                            pipe = pipe_list[i]  # 使用上管道或下管道都可以
                            # 当小鸟通过管道时增加分数
                            if (pipe.rect.right < self.birds[0].rect.left and 
                                pipe.cross_pipe):  # cross_pipe用于确保每个管道只计分一次
                                self.score += 1
                                pipe.cross_pipe = False
                                # 增加通过管道的奖励
                                for g in self.ge:
                                    g.fitness += 5  # 给予额外的适应度奖励

    def update_stats(self, genomes):
        """更新统计信息"""
        try:
            # 计算当前状态
            current_fitness = np.mean([g.fitness for g in self.ge]) if self.ge else 0
            max_fitness = max([g.fitness for g in self.ge]) if self.ge else 0
            
            # 更新统计信息
            stats_text = f"""
******* Running generation {self.generation} *******

Population's average fitness: {current_fitness:.5f}
Best fitness: {max_fitness:.5f}

Population size: {len(self.birds)}
Alive: {len(self.birds)}
Current Score: {self.score}
All-time Best Score: {self.all_time_best_score}

ID    age  size  fitness  adj fit  stag
====  ===  ====  =======  =======  ===="""

            # 添加对齐的数据行
            for i, (genome_id, genome) in enumerate(zip(range(len(self.ge)), self.ge)):
                if i < 5:  # 只显示前5个物种的信息
                    stats_text += f"\n{i:<4}  {0:>3}  {3:>4}  {genome.fitness:>7.1f}  {'-':>7}  {0:>4}"

            # 更新显示
            self.stats_window.update_stats(stats_text)
            self.stats_window.draw(self.window, self.game_surface)
            
        except Exception as e:
            print(f"Error updating stats: {e}")

def pretrain_with_human_data(config):
    """使用人类玩家数据预训练网络"""
    try:
        # 加载所有高分数据（包括备份）
        all_data = []
        main_files = [f for f in os.listdir('.') 
                     if f.startswith('human_play_data_') and f.endswith('.pkl')]
        
        # 处理主文件
        for file in main_files:
            # 从文件名中提取分数
            score = int(file.split('_')[-2])  # 修改这里以获取分数
            if score >= MIN_SCORE_TO_SAVE:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    all_data.extend(data)
                    print(f"Loaded main data with score {score}")
        
        # 处理备份文件
        if os.path.exists(BACKUP_DIR):
            backup_files = [f for f in os.listdir(BACKUP_DIR) 
                          if f.startswith('backup_human_play_data_') and f.endswith('.pkl')]
            for file in backup_files:
                score = int(file.split('_')[-2])  # 修改这里以获取分数
                if score >= MIN_SCORE_TO_SAVE:
                    backup_path = os.path.join(BACKUP_DIR, file)
                    with open(backup_path, 'rb') as f:
                        data = pickle.load(f)
                        all_data.extend(data)
                        print(f"Loaded backup data with score {score}")
        
        print(f"Loaded total {len(all_data)} human play examples")
        
        # 按分数排序，优先使用高分数的数据
        all_data.sort(key=lambda x: x['score'], reverse=True)
        
        # 创建初始种群
        pop = neat.Population(config)
        
        # 对每个基因组进行预训练
        for genome_id, genome in pop.population.items():
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # 使用人类数据训练
            total_error = 0
            for data_point in all_data:
                state = data_point['state']
                action = data_point['action']
                score = data_point['score']
                
                # 获取网络输出
                output = net.activate(state)
                expected = 1 if action == 1 else 0
                
                # 计算误差，高分数据的误差权重更大
                error = (output[0] - expected) ** 2
                weighted_error = error * (1 + score/10)  # 分数越高，权重越大
                total_error += weighted_error
            
            # 设置适应度（误差越小，适应度越高）
            if total_error == 0:
                genome.fitness = 100  # 避免除以零
            else:
                genome.fitness = 1.0 / total_error
        
        # 保存预训练的种群
        with open("pretrained_population.pkl", "wb") as f:
            pickle.dump(pop, f)
            
        return pop
        
    except Exception as e:
        print(f"Error in pretraining: {e}")
        return None

def load_checkpoint(checkpoint_path, config):
    """加载检查点"""
    try:
        # 修改加载方式，设置 weights_only=False
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        
        # 创建新的种群
        pop = neat.Population(config)
        
        # 复制最佳基因组来填充种群
        best_genome = checkpoint['best_genome']
        
        # 为每个个体创建新的基因组
        for i in range(POPULATION_SIZE):
            genome = neat.DefaultGenome(i)
            genome.configure_new(config.genome_config)
            genome.fitness = best_genome.fitness
            # 复制基因组的连接和节点
            for key, conn in best_genome.connections.items():
                genome.connections[key] = conn.copy()
            for key, node in best_genome.nodes.items():
                genome.nodes[key] = node.copy()
            pop.population[i] = genome
            
        pop.generation = checkpoint['generation']
        
        print(f"Loaded checkpoint from generation {checkpoint['generation']}")
        print(f"Best fitness: {checkpoint['best_fitness']}")
        print(f"Score: {checkpoint['score']}")
        print(f"All-time best score: {checkpoint['all_time_best_score']}")
        
        return pop
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def run_neat(config_path):
    """运行NEAT算法"""
    try:
        print(f"正在读取配置文件: {config_path}")
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        # 检查是否有检查点文件
        checkpoint_files = sorted(glob.glob("checkpoints/checkpoint_gen_*.pth"))
        if checkpoint_files:
            # 加载最新的检查点
            pop = load_checkpoint(checkpoint_files[-1], config)
            if pop is None:
                pop = neat.Population(config)
        else:
            # 如果没有检查点，创建新种群
            pop = neat.Population(config)
        
        # 创建游戏实例
        game = Game()
        
        # 添加详细报告器（传入game实例）
        detailed_reporter = DetailedReporter(game)
        pop.add_reporter(detailed_reporter)
        
        # 运行进化过程
        winner = pop.run(game.eval_genomes)
        
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