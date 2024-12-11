CONFIG = {
    "round_limit": 10,
    "data_dir": "./data/start_from_end_1",  # Data Save path
    "model_dir": "./models/start_from_end_1",  # Model Save Path
    "model_file": "./models/start_from_end_1/current_model.pth",  # Current Model file
    "buffer_size": 100000,  # Data Buffer Size
    "max_selfplay_games": 1000,  # Maximum games of self-playing
    "n_playout": 1000,  # MCTS simulation times
    "c_puct": 5,  # MCTS search param
    "batch_size": 64,  # Batch size
    "learning_rate": 1e-3,  # Initial Learning Rate
    "epochs": 10,  # Epoches
    "kl_target": 0.02,  # Goal of KL-Divergence
    "check_freq": 50,  # Frequency of Model Saving
    "game_batch_num": 1000,  # Total Training Batch numes
    "dirichlet_alpha": 0.03,  # Dirichlet Noise Params
    "num_workers": 4,  # Number of Data Collecting Workers
    "use_customized_game": True, #Whether start from a customized state
    "initial_board" : [
    [ 1,  0,  0,  -5,  0,  0,  0,  0,  0], #set the customized state
    [ 0,  1,  0,   0,  0,  0,  0,  0,  0],
    [ 0, 0,  0,   0,  0,  0,  0, 0,  0],
    [0,  0, 0,   0, 0,  0, 0,  0, 0],
    [ 0,  0,  0,   0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,   0,  0,  0,  0,  0,  0],
    [ 0,   0,  0,   0,  0,  0,  0,   0, 0],
    [ 0,  6,   0,   0,  0,  0,  0,   6, 0],
    [ 0,  0,   0,   0,  0,  0,  0,   0, 0],
    [ 0,  0,   0,   0,  5,  0,  0,   0, 0]
    ],
    "initial_player": 1 #set the initial player ofr the customized state

}
