CONFIG = {
    "round_limit": 30,
    "data_dir": "./data/",  # 数据保存路径
    "model_dir": "./models/",  # 模型保存路径
    "model_file": "./models/current_model.pth",  # 当前模型文件
    "buffer_size": 100000,  # 数据缓冲区大小
    "max_selfplay_games": 1000,  # 每次自我对弈的最大局数
    "n_playout": 1000,  # MCTS模拟次数
    "c_puct": 5,  # MCTS探索参数
    "batch_size": 64,  # 训练批次大小
    "learning_rate": 1e-3,  # 初始学习率
    "epochs": 10,  # 每次更新的训练轮数
    "kl_target": 0.02,  # KL散度目标值
    "check_freq": 50,  # 模型保存频率
    "game_batch_num": 1000,  # 总训练批次数
    "dirichlet_alpha": 0.03,  # Dirichlet噪声参数
    "num_workers": 4,  # 数据收集进程数
}
