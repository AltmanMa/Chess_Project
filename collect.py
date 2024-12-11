import os
import pickle
import time
import numpy as np
from collections import deque
from multiprocessing import Pool
from game import GameState
from mcts import MCTSPlayer
from Net import PolicyValueNet
from config import CONFIG
from game import index2move, move2index, flip_map
from filelock import FileLock

class CollectPipeline:
    def __init__(self, init_model=None):
        self.data_buffer = deque(maxlen=CONFIG["buffer_size"])
        self.temp = 1.0  
        self.n_playout = CONFIG["n_playout"]
        self.c_puct = CONFIG["c_puct"]
        self.iters = 0
        self.model_path = init_model or CONFIG["model_file"]

        # Initialize Policy Value Network
        self.policy_value_net = PolicyValueNet(model_file=self.model_path)
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
        )

        # Try to load data from the current data
        self._load_data_buffer()

    def _load_model(self):
        try:
            self.policy_value_net = PolicyValueNet(model_file=self.model_path)
            print("Successfully loaded model.")
        except Exception as e:
            print(f"Failed to load specified model. Using default model. Error: {e}")
            self.policy_value_net = PolicyValueNet()
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
        )



    def _load_data_buffer(self):
        lock_path = CONFIG["data_dir"] + ".lock"
        with FileLock(lock_path):  # 使用文件锁
            if os.path.exists(CONFIG["data_dir"]):
                try:
                    with open(CONFIG["data_dir"] + "/data_buffer.pkl", "rb") as f:
                        data = pickle.load(f)
                        self.data_buffer = data.get("data_buffer", [])
                        self.iters = data.get("iters", 0)
                        print(f"Succesfully loaded data buffer, including {len(self.data_buffer)} samples.")
                except Exception as e:
                    print(f"Failed to load data buffer: {e}. Using empty buffer.")
            else:
                print("Data buffer file not found. Starting with empty buffer.")

    def _save_data_buffer(self):
        lock_path = CONFIG["data_dir"] + ".lock"
        with FileLock(lock_path):  # 使用文件锁
            os.makedirs(CONFIG["data_dir"], exist_ok=True)
            try:
                with open(CONFIG["data_dir"] + "/data_buffer.pkl", "wb") as f:
                    pickle.dump({"data_buffer": self.data_buffer}, f)
                    print("Data buffer saved successfully.")
            except Exception as e:
                print(f"Failed to save data buffer: {e}")
    def get_equi_data(self, play_data):
        """data strengthen"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            print(f"State shape before processing: {state.shape}")
            extend_data.append((state, mcts_prob, winner))
            flipped_state = state[:, :, ::-1]
            flipped_mcts_prob = mcts_prob[::-1]
            extend_data.append((flipped_state, flipped_mcts_prob, winner))
        return extend_data
    
    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            extend_data.append((state, mcts_prob, winner))
            flipped_state = state[:, :, ::-1]
            flipped_mcts_prob = np.zeros_like(mcts_prob)
            for idx, prob in enumerate(mcts_prob):
                flipped_idx = move2index[flip_map(index2move[idx])]
                flipped_mcts_prob[flipped_idx] = prob
            extend_data.append((flipped_state, flipped_mcts_prob, winner))
        return extend_data


    def collect_selfplay_data(self, n_games=1):
        """Collect self-playing data"""
        for i in range(n_games):
            print(f"Starting game {i+1}/{n_games}...")
            winner, play_data = self._self_play_one_game()
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            self.iters += 1
            print(f"Game {i+1} finished. Winner: {winner}. Total iterations: {self.iters}")
        self._save_data_buffer()
        print("Data collection complete.")

    def _self_play_one_game(self):
        """Do one self-play"""
        game = GameState()
        play_data = []
        while True:
            end, winner = game.is_game_over()
            if end:
                play_data = [(state, prob, 1 if winner == game.current_player else -1)
                             for state, prob, _ in play_data]
                return winner, play_data
            move, mcts_probs = self.mcts_player.get_action(game, temp=self.temp, return_prob=True)
            if not move:  
                print("Warning: No valid move returned. Ending game. Current State:")
                print(game.current_state)
                break
            state = game.get_training_state()
            
            play_data.append((state, mcts_probs, None))
            game.make_move(move)


    def run(self, n_games=None):
        try:
            if n_games:
                self.collect_selfplay_data(n_games)
            else:
                while True:
                    self.collect_selfplay_data(1)
        except KeyboardInterrupt:
            print("Interrupted! Saving current data buffer...")
            self._save_data_buffer()
            print("Data saved. Exiting.")

if __name__ == "__main__":
    pipeline = CollectPipeline(init_model=CONFIG["model_file"])
    pipeline.run(n_games=100)
