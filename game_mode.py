import random
import numpy as np
from game import GameState
class GameController:
    def __init__(self, GameState):
        self.board = GameState

    def graphic(self, board):
        """
        可视化棋盘状态
        """
        print(f"当前玩家: {'红方' if board.current_player == 1 else '黑方'}")
        board.render()

    def start_play(self, player1, player2, start_player=1, is_shown=True):
        """
        开始对弈（人机对战或人人对战）
        """
        if start_player not in (1, -1):
            raise ValueError("start_player 应该是 1（红方）或 -1（黑方）")
        self.board.init_board()  # 初始化棋盘
        self.board.current_player = start_player
        players = {1: player1, -1: player2}

        if is_shown:
            self.graphic(self.board)

        while True:
            current_player = self.board.current_player
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.make_move((int(move[0]), int(move[1])), (int(move[2]), int(move[3])))

            if is_shown:
                self.graphic(self.board)

            end, winner = self.board.is_game_over()
            if end:
                if winner == 0:
                    print("游戏结束: 和棋")
                else:
                    print(f"游戏结束: {'红方' if winner == 1 else '黑方'}获胜")
                return winner

    def start_self_play(self, player, is_shown=False, temp=1e-3):
        """
        开始自我对弈并保存训练数据
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []

        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=True)
            states.append(self.board.board.copy())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            self.board.make_move((int(move[0]), int(move[1])), (int(move[2]), int(move[3])))
            if is_shown:
                self.graphic(self.board)

            end, winner = self.board.is_game_over()
            if end:
                winner_z = np.zeros(len(current_players))
                if winner != 0:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0

                player.reset_player()
                return winner, zip(states, mcts_probs, winner_z)

class RandomPlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_action(self, board, temp=None, return_prob=False):
        moves = board.get_all_legal_moves()
        chosen_move = random.choice(moves)
        if return_prob:
            # 构造一个随机概率分布
            probs = np.zeros(len(moves))
            probs[moves.index(chosen_move)] = 1.0
            return chosen_move, probs
        return chosen_move

    def reset_player(self):
        pass  # 在自我对弈中需要重置蒙特卡洛树

if __name__ == "__main__":

    testgame = GameState()
    game_controller = GameController(testgame)

    player1 = RandomPlayer(1)
    player2 = RandomPlayer(-1)

    # 测试人机对弈
    winner = game_controller.start_play(player1, player2, start_player=1, is_shown=True)
    print(f"对弈结束，赢家: {winner}")

    # 测试自我对弈
    mcts_player = RandomPlayer(1)  # 替换为实际的 MCTS 玩家
    winner, training_data = game_controller.start_self_play(mcts_player, is_shown=False)
    print(f"自我对弈结束，赢家: {winner}")