# AI飞小鸟 (AI Flappy Bird)

一个使用NEAT(神经进化增强拓扑)算法实现的AI飞小鸟游戏。AI通过不断学习进化来玩Flappy Bird游戏。

## 功能特点

- 使用NEAT算法实现AI自动学习
- 实时显示训练状态和统计信息
- 可视化神经网络的决策过程
- 支持保存和加载训练模型
- 记录训练历史和最高分
- 支持人类玩家模式

## 项目结构
├── ai_flappy_bird.py # AI训练主程序
├── game.py # 游戏核心逻辑
├── bird.py # 小鸟
├── pipe.py # 管道
├── StatsWindow.py # 统计窗口
├── DetailedReporter.py # 训练数据记录器
├── config-feedforward.txt # NEAT配置文件
├── 微软正黑体.ttf # 显示字体
│
├── img/ # 图片资源
│ ├── bg.png # 背景
│ ├── bird1.png # 小鸟动画
│ ├── bird2.png # 小鸟动画
│ ├── ground.png # 地面
│ └── pipe.png # 管道
│
└── training_data/ # 训练数据
├── scores_history.json # 分数历史
└── checkpoints/ # 模型检查点

## 安装依赖
bash
pip install pygame
pip install neat-python
pip install numpy
pip install torch

## 使用方法

1. 运行AI训练:
bash
py ai_flappy_bird.py

2. 运行人类玩家模式:
bash
py ai_flappy_bird.py --human

## 实现细节

- 使用NEAT算法进行神经网络训练
- 输入层包含:
  - 小鸟到管道的水平距离
  - 小鸟到管道的垂直距离
  - 小鸟的下落速度
- 输出层决定是否跳跃
- 使用适应度函数评估AI表现

## 许可证

MIT License

## 作者

lucker3900