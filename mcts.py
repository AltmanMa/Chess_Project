import numpy as np
import copy
from config import CONFIG
from game import index2move, move2index

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} 
        self._n_visits = 0 
        self._Q = 0 
        self._u = 0 
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    """
    MCTS
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """
        :param policy_value_fn: return probablities of moves and their value
        :param c_puct: hyper params to balance exploration 
        :param n_playout: Numer of playouts
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        Do one simulation and back paprogation updating the treenode
        """
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.make_move(action)
            end, winner = state.is_game_over()
            if end:
                leaf_value = 0.0 if winner == 0 else (1.0 if winner == state.current_player else -1.0)
                node.update_recursive(-leaf_value)
                return

        # Use Policy Value Network to value the node
        action_probs, leaf_value = self._policy(state)
        if not action_probs:  # No legal moves to expand
            end, winner = state.is_game_over()
            if end:
                print(f'game is over at round {state.round_count}')
            else:
                print("No legal moves in playout.")
            print(f'current Player is: {state.current_player}')
            return
        end, winner = state.is_game_over()

        if not end:
            node.expand(action_probs)
        else:
            leaf_value = 0.0 if winner == 0 else (1.0 if winner == state.current_player else -1.0)

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        Get All legal moves and their probabilities
        :param state: Current Game State
        :param temp: Temp Param to control how much to search
        """
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        if not self._root._children:
            print("No children nodes in root. Possible issue with expansion or playout.")
            print("Current state:")
            print(state.current_state)
            print(f'Last move is {state.last_move}:')
            print(f'current round is {state.round_count}')
            print(f'cuurent player is: {state.current_player}')
            return [], []

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        if not act_visits:
            print("No visits recorded for any actions.")
            return [], []

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSPlayer:
    """
    MCTS AI Player
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
        print(probs)

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
