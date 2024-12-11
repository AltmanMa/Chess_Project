import os
import random
import numpy as np
import pickle
import time
from Net import PolicyValueNet
from config import CONFIG

class TrainPipeline:
    def __init__(self, init_model=None):
        self.learn_rate = CONFIG["learning_rate"]
        self.lr_multiplier = 1.0  # modify learning rate
        self.batch_size = CONFIG["batch_size"]
        self.epochs = CONFIG["epochs"]
        self.kl_targ = CONFIG["kl_target"]
        self.check_freq = CONFIG["check_freq"]
        self.game_batch_num = CONFIG["game_batch_num"]

        self.data_buffer = []
        self.buffer_path = CONFIG["data_dir"] + "/data_buffer.pkl"

        if init_model and os.path.exists(init_model):
            self.policy_value_net = PolicyValueNet(model_file=init_model)
            print("Successfully Load Model")
        else:
            self.policy_value_net = PolicyValueNet()
            print("No initial Model Found. Start from beginning")

    def _load_data_buffer(self):
        """Load Data buffer"""
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path, "rb") as f:
                data = pickle.load(f)
                self.data_buffer = data["data_buffer"]
                print(f"Succesfully loaded data buff, inluding numbers of samples: {len(self.data_buffer)}")
        else:
            print("Havn't find data buffer, empty buffer used")

    def _incremental_load_data_buffer(self):
        """增量加载数据缓冲区"""
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path, "rb") as f:
                data = pickle.load(f)
                new_data = data["data_buffer"]
                if len(new_data) > len(self.data_buffer):  # 如果有新数据
                    self.data_buffer.extend(new_data[len(self.data_buffer):])
                    print(f"New data loaded, buffer size updated to {len(self.data_buffer)}")
                else:
                    print("No new data found in buffer.")
        else:
            print("No data buffer found.")

    def _save_data_buffer(self):
        os.makedirs(CONFIG["data_dir"], exist_ok=True)
        with open(self.buffer_path, "wb") as f:
            pickle.dump({"data_buffer": self.data_buffer}, f)
        print("Data Buffer Saved")

    def policy_update(self):
        """Upadata Policy Value network"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch], dtype="float32")
        mcts_probs_batch = np.array([data[1] for data in mini_batch], dtype="float32")
        winner_batch = np.array([data[2] for data in mini_batch], dtype="float32")

        old_probs, _ = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch
            )
            new_probs, _ = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(
                    old_probs
                    * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  
                print("KL excceed limit, stop training")
                break

        # ADjust learning rate dinamiclly
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(f"KL Divergence: {kl:.5f}, Learning rate Modifying rate: {self.lr_multiplier:.3f}, Loss: {loss}, Entropy: {entropy}")
        return loss, entropy

    def run(self):
        self._load_data_buffer()
        for i in range(self.game_batch_num):
            print(f"Traning Number {i + 1} started")
            print(f"Buffer size: {len(self.data_buffer)}")
            self._incremental_load_data_buffer()
            if len(self.data_buffer) < self.batch_size:
                print("No enough data, Wating for data collection...")
                time.sleep(30)  #
                continue

            # Do one policy update
            self.policy_update()

            # Save model
            self.policy_value_net.save_model(CONFIG["model_file"])
            if (i + 1) % self.check_freq == 0:
                checkpoint_path = CONFIG["model_dir"] + f"/policy_batch_{i + 1}.model"
                self.policy_value_net.save_model(checkpoint_path)
                print(f"Model Saved to {checkpoint_path}")


if __name__ == "__main__":
    trainer = TrainPipeline(init_model=CONFIG["model_file"])
    trainer.run()
