import neat
import numpy as np
import time

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