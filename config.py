CONFIG = {
    "round_limit" : 30,
    "data_dir": "./data/",  # save
    "model_dir": "./models/",  # 模型保存路径
    "model_file": "./models/current_model.pth",  # 当前模型文件
    'buffer_size': 100000,
    "max_selfplay_games": 1000,  # 每次自我对弈的最大局数
    "n_playout": 1000,  # MCTS的模拟次数
    "c_puct": 5,  # MCTS探索参数
    "batch_size": 64,  # 训练批次大小
    "lr": 1e-3,  # 初始学习率
    "num_epochs": 10,  # 训练轮数
    "dirichlet_alpha": 0.03,  # Dirichlet噪声参数
    "num_workers": 4,  # 数据收集进程数
}
