import pygame
import numpy as np
import os
import json
import torch
import logging
import random
import neat
from bird import Bird  # 直接导入
from pipe import Pipe  # 直接导入
from StatsWindow import StatsWindow  # 直接导入

# 游戏窗口设置
SCREEN_WIDTH = 780
SCREEN_HEIGHT = 600
FPS = 60
PIPE_FREQUENCY = 1500  # 管道生成频率
POPULATION_SIZE = 30  # 添加种群大小常量
STATS_UPDATE_INTERVAL = 2  # 统计信息更新间隔（帧数）
MAX_PIPE_PAIRS = 3  # 最大管道对数
DRAW_DEBUG = True  # 添加调试模式常量

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.font.init()  # 初始化字体模块
        
        # 创建一个更宽的窗口来容纳游戏和统计信息
        self.window = pygame.display.set_mode((SCREEN_WIDTH + 400, SCREEN_HEIGHT))
        pygame.display.set_caption("CC飞小鸟")
        
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
        self.model_dir = os.path.join("training_data", "checkpoints")  # 修改保存目录
        
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
        
        # 绘制主射线（红色线 - 从小鸟到管道中心）
        pygame.draw.line(
            self.game_surface,
            (255, 0, 0),  # 红色
            bird_pos,
            target_pos,
            2  # 线宽
        )
        
        # 绘制管道空隙中心的标记（绿色线）
        if len(pipes) >= 2:
            for i in range(0, len(pipes), 2):
                if i + 1 < len(pipes):
                    top_pipe = pipes[i] if pipes[i].is_top else pipes[i+1]
                    bottom_pipe = pipes[i+1] if not pipes[i+1].is_top else pipes[i]
                    
                    # 计算空隙中间点
                    gap_center_x = top_pipe.rect.centerx
                    gap_center_y = (top_pipe.rect.bottom + bottom_pipe.rect.top) / 2 + 70

    def draw_debug_info(self, bird, pipes):
        """移除调试信息绘制"""
        pass  # 不绘制任何调试信息

    def load_best_score(self):
        """加载历史最佳分数"""
        try:
            score_path = os.path.join('training_data', 'best_score.txt')
            if os.path.exists(score_path):
                with open(score_path, 'r') as f:
                    self.all_time_best_score = int(f.read())
                print(f"Loaded all-time best score: {self.all_time_best_score}")
        except Exception as e:
            print(f"Error loading best score: {e}")
            
    def save_best_score(self):
        """保存历史最佳分数"""
        try:
            score_path = os.path.join('training_data', 'best_score.txt')
            with open(score_path, 'w') as f:
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
            history_path = os.path.join('training_data', 'scores_history.json')
            with open(history_path, 'w') as f:
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
                    'training_data', 'checkpoints',
                    f'checkpoint_gen_{generation}.pth'
                )
                torch.save(checkpoint, checkpoint_path)
            
            # 如果是最佳分数，单独保存
            if self.score > self.best_score:
                self.best_score = self.score
                best_model_path = os.path.join(
                    'training_data', 'checkpoints',
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
                
                # 绘制管道和边框
                self.pipe_sprite.draw(self.game_surface)
                for pipe in self.pipe_sprite:
                    # 绘制管道的红色边框
                    pygame.draw.rect(self.game_surface, (255, 0, 0), pipe.rect, 2)
                
                # 绘制小鸟和边框
                for bird in self.birds:
                    self.game_surface.blit(bird.image, bird.rect)
                    # 绘制小鸟的红色边框
                    pygame.draw.rect(self.game_surface, (255, 0, 0), bird.rect, 2)
                    
                    # 如果启用了调试模式,绘制传感器线
                    if DRAW_DEBUG:
                        self.draw_sensors(bird, self.pipe_sprite.sprites())
                
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