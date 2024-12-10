import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os
from game import index2move, move2index
# 配置参数
ACTION_SPACE = 2086  # 动作空间大小（假设有2086个合法动作）
INPUT_CHANNELS = 9  # 输入通道数
BOARD_HEIGHT = 10  # 棋盘高度
BOARD_WIDTH = 9  # 棋盘宽度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 残差块定义
class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        return self.relu(x)


# 策略价值网络
class Net(nn.Module):
    def __init__(self, num_channels=256, num_res_blocks=7):
        super(Net, self).__init__()
        # 公共特征提取
        self.conv_block = nn.Conv2d(INPUT_CHANNELS, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])
        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * BOARD_HEIGHT * BOARD_WIDTH, ACTION_SPACE)
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * BOARD_HEIGHT * BOARD_WIDTH, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 公共特征
        x = self.conv_block(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.relu(value)
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = self.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)
        return policy, value


# 策略价值网络的封装类
class PolicyValueNet:
    def __init__(self, model_file=None, lr=0.002, use_gpu=True):
        self.device = DEVICE
        self.net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = GradScaler()
        self.lr = lr
        if model_file:
            self.load_model(model_file)

    def policy_value(self, state_batch):
        """输入一批状态，输出动作概率和局面价值"""
        self.net.eval()
        with torch.no_grad():
            state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
            log_act_probs, value = self.net(state_batch)
            act_probs = np.exp(log_act_probs.cpu().numpy())
            return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """输入棋盘状态，返回合法动作概率及局面价值"""
        self.net.eval()
        legal_moves = board.get_all_legal_moves()
        current_state = board.get_training_state()
        current_state = current_state[np.newaxis, :] 
        current_state = torch.tensor(current_state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            log_act_probs, value = self.net(current_state)
        act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
        legal_act_probs = [(move, act_probs[move2index[move]]) for move in legal_moves]
        return legal_act_probs, value.item()

    def train_step(self, state_batch, mcts_probs, winner_batch):
        """训练一步"""
        self.net.train()
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32).to(self.device)
        winner_batch = torch.tensor(winner_batch, dtype=torch.float32).to(self.device)
        self.optimizer.zero_grad()
        with autocast():
            log_act_probs, value = self.net(state_batch)
            value_loss = F.mse_loss(value.view(-1), winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
            loss = value_loss + policy_loss
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def adjust_lr(self, lr):
        """动态调整学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_model(self, model_file):
        torch.save(self.net.state_dict(), model_file)

    def load_model(self, model_file):
        if os.path.exists(model_file):
            self.net.load_state_dict(torch.load(model_file, map_location=self.device))
            print(f"模型加载成功：{model_file}")
        else:
            print(f"模型文件未找到：{model_file}，使用初始模型")



if __name__ == '__main__':
    # 测试网络
    net = Net().to(DEVICE)
    dummy_input = torch.ones((8, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)).to(DEVICE)
    policy, value = net(dummy_input)
    print(f"策略输出形状: {policy.shape}, 价值输出形状: {value.shape}")