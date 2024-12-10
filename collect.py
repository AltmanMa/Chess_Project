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

    def _load_data_buffer(self):
        """load the existing data"""
        if os.path.exists(CONFIG["data_dir"]):
            try:
                with open(CONFIG["data_dir"] + "/data_buffer.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.data_buffer = data["data_buffer"]
                    self.iters = data["iters"]
                    print(f"Successfully load data, current iteration: {self.iters}")
            except Exception as e:
                print(f"Fail to load data: {e}")

    def _save_data_buffer(self):
        """Save the data"""
        os.makedirs(CONFIG["data_dir"], exist_ok=True)
        with open(CONFIG["data_dir"] + "/data_buffer.pkl", "wb") as f:
            pickle.dump({"data_buffer": self.data_buffer, "iters": self.iters}, f)

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

    def collect_selfplay_data(self, n_games=1):
        """Collect self-playing data"""
        print("Start Collecting Data by self-playing...")
        for i in range(n_games):
            winner, play_data = self._self_play_one_game()
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            self.iters += 1
            print(f"Game {i+1}/{n_games} finished. Total episodes: {self.iters}")

        self._save_data_buffer()

    def _self_play_one_game(self):
        """Do one self-play"""
        game = GameState()
        play_data = []
        while True:
            move, mcts_probs = self.mcts_player.get_action(game, temp=self.temp, return_prob=True)
            if not move:  
                print("Warning: No valid move returned. Ending game. Current State:")
                print(game.current_state)
                break
            state = game.get_training_state()
            end, winner = game.is_game_over()
            if end:
                play_data = [(state, prob, 1 if winner == game.current_player else -1)
                             for state, prob, _ in play_data]
                return winner, play_data
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
            print("Data Collection Ended. Saving to data...")
            self._save_data_buffer()


if __name__ == "__main__":
    pipeline = CollectPipeline(init_model=CONFIG["model_file"])
    pipeline.run(n_games=100)
