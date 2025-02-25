import pygame
from pygame.midi import frequency_to_midi
from pipe import Pipe
from bird import Bird
import random
import os
import sys
import traceback
import logging
import numpy as np
import pickle
import time

# 设置日志
logging.basicConfig(
    filename='game_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, relative_path)
    except Exception as e:
        logging.error(f"Error in resource_path: {e}")
        logging.error(traceback.format_exc())
        return relative_path


# 游戏窗口设置
SCREEN_WIDTH = 780
SCREEN_HEIGHT = 600
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

logging.info("Initializing pygame")
pygame.init()
pygame.mixer.init()

# 加载资源文件

bg_img = pygame.image.load(resource_path("img/bg.png"))

bg_img = pygame.transform.scale(bg_img, (780, 600))
ground_img = pygame.image.load(resource_path("img/ground.png"))
pipe_btm_img = pygame.image.load(resource_path("img/pipe.png"))
restart_img = pygame.image.load(resource_path("img/restart.png"))
pause_img = pygame.image.load(resource_path("img/pause.png"))

# 加载音效
logging.info("Loading sound files")
bg_music = pygame.mixer.Sound(resource_path("sound/bgm.wav"))

# 加载小鸟图片
logging.info("Loading bird images")
bird_imgs = []
for i in range(1, 3):
    img_path = resource_path(f"img/bird{i}.png")
    logging.info(f"Loading {img_path}")
    bird_imgs.append(pygame.image.load(img_path))

# 设置音量
bg_music.set_volume(0.3)

# 播放背景音乐
bg_music.play(-1)

clock = pygame.time.Clock()
run = True
x = 0
y = 0
gd_x = 0
pipe_x = 200
pipe_speed = 4
pipe_distance = 150

pipes_top = []
pipes_btm = []

window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#bird1_img = pygame.image.load("img/bird1.png")
#bird_sprite = pygame.sprite.Group()

pipe_top_img = pygame.transform.flip(pipe_btm_img, False, True)
#旋转管子180度
#pipe_top = Pipe(SCREEN_WIDTH, SCREEN_HEIGHT/2 - pipe_distance/2, pipe_top_img, True)

pipe_sprite = pygame.sprite.Group()

#pipe_sprite.add(pipe_btm)
#pipe_sprite.add(pipe_top)

bird = Bird(100, SCREEN_HEIGHT/2, bird_imgs)
bird.set_ground_height(SCREEN_HEIGHT - 100)  # 设置地面高度

bird_sprite = pygame.sprite.Group()
bird_sprite.add(bird)

pygame.display.set_caption("cc飞小鸟")
pygame.display.set_icon(bird_imgs[0])

#分数记录
game_score = 0

font = pygame.font.Font(resource_path("微软正黑体.ttf"), 30)

frequency = 1500
last_update = pygame.time.get_ticks() - frequency
pause_time = 0  # 添加暂停时的时间记录

game_over = False
game_pause = False

training_data = []
last_state = None
last_action = None

# 添加新的常量
MIN_SCORE_TO_SAVE = 100  # 只保存达到这个分数的游戏数据
MAX_TRAINING_FILES = 10  # 最多保存10个高分记录
BACKUP_DIR = "training_data_backup"  # 备份目录

def get_game_state(bird, pipes):
    """获取当前游戏状态"""
    if not pipes:
        return np.zeros(4)
        
    next_pipe = None
    for pipe in pipes:
        if pipe.rect.right > bird.rect.left and pipe.cross_pipe:  # 使用cross_pipe替代passed
            next_pipe = pipe
            break
            
    if not next_pipe:
        return np.zeros(4)
        
    return np.array([
        bird.rect.y / SCREEN_HEIGHT,  # 小鸟高度
        bird.down_speed / 10,  # 小鸟速度
        (next_pipe.rect.x - bird.rect.x) / SCREEN_WIDTH,  # 水平距离
        (next_pipe.rect.y - bird.rect.y) / SCREEN_HEIGHT  # 垂直距离
    ])

def save_training_data(training_data, score):
    """保存训练数据,包含备份机制"""
    try:
        if score >= MIN_SCORE_TO_SAVE:
            # 添加时间戳到文件名
            timestamp = int(time.time())
            filename = f"human_play_data_{score}_{timestamp}.pkl"
            backup_filename = os.path.join(BACKUP_DIR, f"backup_human_play_data_{score}_{timestamp}.pkl")
            
            # 打印当前所有训练数据文件
            existing_files = [f for f in os.listdir('.') 
                            if f.startswith('human_play_data_') and f.endswith('.pkl')]
            print(f"\nCurrent training files: {existing_files}")
            
            if len(existing_files) >= MAX_TRAINING_FILES:
                # 找到分数最低的文件
                lowest_score_file = min(existing_files, 
                    key=lambda x: int(x.split('_')[-2]))  # 修改这里以获取分数
                lowest_score = int(lowest_score_file.split('_')[-2])
                
                # 只有当新分数更高时才替换
                if score > lowest_score:
                    os.remove(lowest_score_file)
                    print(f"Removed {lowest_score_file} (score: {lowest_score}) to make room for new data")
                else:
                    print(f"New score ({score}) not higher than lowest existing score ({lowest_score}), skipping save")
                    return
            
            # 保存主文件
            print(f"Saving training data to {filename}")
            with open(filename, "wb") as f:
                pickle.dump(training_data, f)
            
            # 保存备份文件
            print(f"Creating backup at {backup_filename}")
            with open(backup_filename, "wb") as f:
                pickle.dump(training_data, f)
            
            print(f"Successfully saved training data with score {score}")
            
            # 打印保存后的文件列表
            updated_files = [f for f in os.listdir('.') 
                           if f.startswith('human_play_data_') and f.endswith('.pkl')]
            print(f"Updated training files: {updated_files}\n")
            
    except Exception as e:
        print(f"Error saving training data: {e}")
        # 尝试紧急备份
        try:
            emergency_backup = f"emergency_backup_{score}_{int(time.time())}.pkl"
            with open(emergency_backup, "wb") as f:
                pickle.dump(training_data, f)
            print(f"Created emergency backup: {emergency_backup}")
        except Exception as e2:
            print(f"Failed to create emergency backup: {e2}")

while run:
    clock.tick(FPS)

    #游戏输入
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            save_training_data(training_data, game_score)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if game_over and not game_pause:  # 只在非暂停状态下响应点击
                    # 重置游戏
                    game_over = False
                    game_score = 0
                    bird.rect.center = (100, SCREEN_HEIGHT/2)
                    bird.down_speed = 0
                    pipe_sprite.empty()  # 清空所有管道
                    last_update = pygame.time.get_ticks() - frequency  # 重置管道生成时间
                    bg_music.play(-1)  # 重新播放背景音乐
                elif not game_pause:  # 只在非暂停状态下响应点击
                    current_state = get_game_state(bird, pipe_sprite.sprites())
                    training_data.append({
                        'state': current_state,
                        'action': 1,  # 跳跃
                        'score': game_score
                    })
                    bird.jump(True)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and not game_over:
                bird.jump(True)
            elif event.key == pygame.K_SPACE and game_over:
                # 重置游戏
                game_over = False
                game_score = 0
                bird.rect.center = (100, SCREEN_HEIGHT/2)
                bird.down_speed = 0
                pipe_sprite.empty()  # 清空所有管道
                last_update = pygame.time.get_ticks() - frequency  # 重置管道生成时间
                bg_music.play(-1)  # 重新播放背景音乐
            elif event.key == pygame.K_SPACE and not game_over and not game_pause:
                current_state = get_game_state(bird, pipe_sprite.sprites())
                training_data.append({
                    'state': current_state,
                    'action': 1,  # 跳跃
                    'score': game_score
                })
                bird.jump(True)
            elif event.key == pygame.K_SPACE and not game_over and game_pause:
                game_pause = False
                bg_music.play(-1)
                pygame.mouse.set_visible(True)
                # 恢复时，更新last_update，加上暂停的时间差
                current_time = pygame.time.get_ticks()
                time_diff = current_time - pause_time
                last_update += time_diff

    if (pygame.mouse.get_pressed()[0] or pygame.key.get_pressed()[pygame.K_UP]) and not game_over and not game_pause:  # 添加not game_pause条件
            bird.jump(False)

    #游戏结束
    if pygame.sprite.groupcollide(bird_sprite, pipe_sprite, False, False)\
       or bird.rect.bottom >= SCREEN_HEIGHT -100\
       or bird.rect.top <= 0:
        #if not game_over:  # 只在第一次碰撞时停止音乐
        bg_music.stop()
        game_over = True

    #游戏更新
    if not game_over and not game_pause:
        bird_sprite.update()
        pipe_sprite.update()
        #碰撞判断
        now = pygame.time.get_ticks()


        if now - last_update > frequency:
            # 生成一个管子
            pipe_y = random.randint(-100, 100)
            pipe_btm = Pipe(SCREEN_WIDTH, SCREEN_HEIGHT / 2 + pipe_distance/2 + pipe_y, pipe_btm_img, False)
            pipe_top = Pipe(SCREEN_WIDTH, SCREEN_HEIGHT / 2 - pipe_distance/2 + pipe_y, pipe_top_img, True)
            #pipe_sprite = pygame.sprite.Group()

            pipe_sprite.add(pipe_btm)
            pipe_sprite.add(pipe_top)
            last_update = now

        pipes = pipe_sprite.sprites()
        if len(pipes) > 0:
            first_pipe = pipes[0]
            if bird.rect.left > first_pipe.rect.right and first_pipe.cross_pipe:
                game_score += 1
                first_pipe.cross_pipe = False

        # 游戏显示
        window.blit(bg_img, (0,0))
        #window.draw.rect(groun, (255, 0, 0), (900, SCREEN_HEIGHT - 100, 100, 100))
        gd_x -= 4
        if(gd_x <= -100):
            gd_x = 0

        if not pygame.key.get_pressed()[pygame.K_SPACE]:
            current_state = get_game_state(bird, pipe_sprite.sprites())
            training_data.append({
                'state': current_state,
                'action': 0,  # 不跳跃
                'score': game_score
            })
    elif game_over:
        bird.drop()

        if len(training_data) > 0:
            save_training_data(training_data, game_score)
            training_data = []

    window.blit(bg_img, (0, 0))
    bird_sprite.draw(window)
    pipe_sprite.draw(window)
    window.blit(ground_img, (gd_x, SCREEN_HEIGHT - 100))

    #显示分数
    cc_text = font.render("CC无聊纯娱乐", True, (0, 0, 255))
    window.blit(cc_text, (SCREEN_WIDTH / 2 - 100, 0))
    score_text = font.render(f"得分:{str(game_score)}", True, (0, 0, 255))
    window.blit(score_text, (SCREEN_WIDTH/2 - 50, 50))

    if game_over:
        # 获取重启按钮的矩形区域并设置中心点
        restart_rect = restart_img.get_rect()
        restart_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2)  # 设置为屏幕中心
        window.blit(restart_img, restart_rect)  # 使用rect来定位

    if game_pause:
        # 获取暂停按钮的矩形区域并设置中心点
        pause_rect = pause_img.get_rect()
        pause_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2)  # 设置为屏幕中心
        window.blit(pause_img, pause_rect)  # 使用rect来定位

    pygame.display.update()

pygame.quit()