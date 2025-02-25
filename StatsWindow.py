import pygame

SCREEN_WIDTH = 780  # 直接在文件中定义常量

class StatsWindow:
    def __init__(self):
        # 使用两种大小的字体
        self.font = pygame.font.Font("微软正黑体.ttf", 16)  # 统计信息用小字体
        self.score_font = pygame.font.Font("微软正黑体.ttf", 30)  # 标题和分数用大字体
        self.stats_text = ""
        self.history_text = []  # 存储历史记录
        
    def draw(self, window, game_surface):
        """绘制统计信息"""
        # 在主窗口上绘制游戏画面和统计信息
        window.fill((50, 50, 50))  # 深灰色背景
        window.blit(game_surface, (0, 0))
        
        # 从stats_text中提取当前分数
        current_score = "0"
        for line in self.stats_text.split('\n'):
            if "Current Score:" in line:
                current_score = line.split(":")[1].strip()
                break
        
        #显示分数（使用大号字体并居中）
        cc_text = self.score_font.render("CC无聊纯娱乐", True, (0, 0, 255))
        score_text = self.score_font.render(f"得分:{current_score}", True, (0, 0, 255))
        
        # 计算文本位置使其居中
        cc_text_width = cc_text.get_width()
        score_text_width = score_text.get_width()
        
        cc_x = (SCREEN_WIDTH - cc_text_width) // 2
        score_x = (SCREEN_WIDTH - score_text_width) // 2
        
        # 绘制居中的文本
        window.blit(cc_text, (cc_x, 0))
        window.blit(score_text, (score_x, 50))

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