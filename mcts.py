import numpy as np
import copy
from config import CONFIG
from game import index2move, move2index

def softmax(x):
    """计算softmax值"""
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """
    蒙特卡洛树搜索中的节点类
    跟踪访问次数（N），动作价值（Q），置信上限调整值（U），以及先验概率（P）
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 映射动作到子节点
        self._n_visits = 0   # 节点访问次数
        self._Q = 0          # 平均动作价值
        self._u = 0          # UCB1置信上限
        self._P = prior_p    # 先验概率

    def expand(self, action_priors):
        """扩展叶节点，创建子节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        根据PUCT公式选择具有最大Q+U值的子节点
        返回(action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """计算当前节点的Q+U值"""
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """
        更新节点的Q值和访问次数
        leaf_value: 当前节点的叶子节点的价值
        """
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        对当前节点及其所有祖先节点递归更新
        """
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """判断节点是否为叶子节点"""
        return self._children == {}

    def is_root(self):
        """判断节点是否为根节点"""
        return self._parent is None


class MCTS:
    """
    蒙特卡洛树搜索实现
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """
        :param policy_value_fn: 函数，输入状态，返回动作概率和盘面价值
        :param c_puct: 控制探索和利用之间平衡的超参数
        :param n_playout: 每次搜索的模拟次数
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        进行一次模拟，基于叶子节点的评估值反向更新整棵树
        """
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.make_move(action)

        # 使用策略值网络评估叶子节点
        action_probs, leaf_value = self._policy(state)
        end, winner = state.is_game_over()

        if not end:
            node.expand(action_probs)
        else:
            leaf_value = 0.0 if winner == 0 else (1.0 if winner == state.current_player else -1.0)

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        返回所有合法动作及其对应的概率
        :param state: 当前游戏状态
        :param temp: 温度参数，控制探索程度
        """
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        更新根节点为当前动作对应的子节点，保持子树状态
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSPlayer:
    """
    基于MCTS的AI玩家
    """

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        move_probs = np.zeros(2086)
        acts, probs = self.mcts.get_move_probs(board, temp)
        indices = [move2index[act] for act in acts]
        move_probs[indices] = probs

        if self._is_selfplay:
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(CONFIG['dirichlet_alpha'] * np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        else:
            return move
