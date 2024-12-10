import os
import random
import numpy as np
import pickle
import time
from Net import PolicyValueNet
from config import CONFIG

class TrainPipeline:
    def __init__(self, init_model=None):
        # 训练参数
        self.learn_rate = CONFIG["learning_rate"]
        self.lr_multiplier = 1.0  # 动态调整学习率
        self.batch_size = CONFIG["batch_size"]
        self.epochs = CONFIG["epochs"]
        self.kl_targ = CONFIG["kl_target"]
        self.check_freq = CONFIG["check_freq"]
        self.game_batch_num = CONFIG["game_batch_num"]

        # 初始化数据缓冲区
        self.data_buffer = []
        self.buffer_path = CONFIG["data_dir"] + "/data_buffer.pkl"

        # 加载模型
        if init_model and os.path.exists(init_model):
            self.policy_value_net = PolicyValueNet(model_file=init_model)
            print("已加载初始模型")
        else:
            self.policy_value_net = PolicyValueNet()
            print("未找到初始模型，从零开始训练")

    def _load_data_buffer(self):
        """加载数据缓冲区"""
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path, "rb") as f:
                data = pickle.load(f)
                self.data_buffer = data["data_buffer"]
                print(f"数据缓冲区加载成功，包含样本数: {len(self.data_buffer)}")
        else:
            print("未找到数据缓冲区，使用空缓冲区")

    def _save_data_buffer(self):
        """保存数据缓冲区"""
        os.makedirs(CONFIG["data_dir"], exist_ok=True)
        with open(self.buffer_path, "wb") as f:
            pickle.dump({"data_buffer": self.data_buffer}, f)
        print("数据缓冲区已保存")

    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch], dtype="float32")
        mcts_probs_batch = np.array([data[1] for data in mini_batch], dtype="float32")
        winner_batch = np.array([data[2] for data in mini_batch], dtype="float32")

        # 旧策略分布
        old_probs, _ = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )
            # 新策略分布
            new_probs, _ = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(
                    old_probs
                    * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  # KL 散度过大时提前终止
                print("KL 散度超出阈值，提前终止训练")
                break

        # 动态调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(f"KL 散度: {kl:.5f}, 学习率调整系数: {self.lr_multiplier:.3f}, 损失: {loss}, 熵: {entropy}")
        return loss, entropy

    def run(self):
        """开始训练"""
        self._load_data_buffer()
        for i in range(self.game_batch_num):
            print(f"开始第 {i + 1} 次训练")
            if len(self.data_buffer) < self.batch_size:
                print("数据不足，等待更多自我对弈数据...")
                time.sleep(30)  # 等待数据生成
                continue

            # 执行一次策略更新
            self.policy_update()

            # 保存模型
            self.policy_value_net.save_model(CONFIG["model_file"])
            if (i + 1) % self.check_freq == 0:
                checkpoint_path = CONFIG["model_dir"] + f"/policy_batch_{i + 1}.model"
                self.policy_value_net.save_model(checkpoint_path)
                print(f"模型已保存至 {checkpoint_path}")


if __name__ == "__main__":
    trainer = TrainPipeline(init_model=CONFIG["model_file"])
    trainer.run()
