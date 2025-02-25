import pygame
import neat
import numpy as np
import random
import os
import sys
import logging
import pickle
import time
import torch
from game import Game  # 直接导入，不使用models.
from DetailedReporter import DetailedReporter  # 直接导入
from StatsWindow import StatsWindow  # 直接导入

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

def pretrain_with_human_data(config):
    """使用人类玩家数据预训练网络"""
    try:
        # 加载所有高分数据
        all_data = []
        human_data_dir = os.path.join('training_data', 'human_play_data')
        
        if os.path.exists(human_data_dir):
            main_files = [f for f in os.listdir(human_data_dir) 
                         if f.startswith('human_play_data_') and f.endswith('.pkl')]
            
            # 处理主文件
            for file in main_files:
                try:
                    score = int(file.split('_')[-2])  # 获取分数
                    if score >= MIN_SCORE_TO_SAVE:
                        file_path = os.path.join(human_data_dir, file)
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            all_data.extend(data)
                            print(f"Loaded human play data with score {score}")
                except Exception as e:
                    print(f"Error loading file {file}: {e}")
                    continue
        
        if not all_data:
            print("No human play data found")
            return neat.Population(config)
            
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
        
        # 读取配置文件
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        # 尝试加载最佳检查点
        checkpoint_path = os.path.join('training_data', 'checkpoints', 'best_model_score_55.pth')
        if os.path.exists(checkpoint_path):
            print(f"找到检查点文件: {checkpoint_path}")
            pop = load_checkpoint(checkpoint_path, config)
            if pop is None:
                print("加载检查点失败,创建新种群")
                pop = neat.Population(config)
        else:
            print("未找到检查点文件,创建新种群")
            pop = neat.Population(config)
        
        # 创建游戏实例
        game = Game()
        
        # 添加详细报告器
        detailed_reporter = DetailedReporter(game)
        pop.add_reporter(detailed_reporter)
        
        # 运行进化过程
        winner = pop.run(game.eval_genomes)
        
        # 保存最佳网络
        with open("best.pickle", "wb") as f:
            pickle.dump(winner, f)
            
    except Exception as e:
        print(f"发生错误: {e}")
        logging.error(f"Error in run_neat: {e}")
        raise

def ensure_directories():
    """确保所有必要的目录都存在"""
    directories = [
        'training_data',
        os.path.join('training_data', 'checkpoints'),
        os.path.join('training_data', 'human_play_data'),
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

if __name__ == "__main__":
    # 确保目录存在
    ensure_directories()
    
    # 获取配置文件的绝对路径
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    
    # 确保配置文件存在
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
        
    try:
        run_neat(config_path)
    except Exception as e:
        print(f"错误: {e}") 