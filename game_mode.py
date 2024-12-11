import random
import numpy as np
from game import GameState
from mcts import MCTSPlayer
from Net import PolicyValueNet
from config import CONFIG
class GameController:
    def __init__(self, GameState):
        self.board = GameState

    def graphic(self, board):
        """
        Visulize the borad
        """
        print(f"Current Player: {'Red' if board.current_player == 1 else 'Black'}")
        board.render()

    def start_play(self, player1, player2, start_player=1, is_shown=True):
        if start_player not in (1, -1):
            raise ValueError("start_player should among 1 (red) or -1 (black)")
        self.board.init_board() 
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
                    print("Game end: Draw")
                else:
                    print(f"Game End: {'Red' if winner == 1 else 'Black'} wins")
                return winner

    def start_self_play(self, player, is_shown=False, temp=1e-3):
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
            probs = np.zeros(len(moves))
            probs[moves.index(chosen_move)] = 1.0
            return chosen_move, probs
        return chosen_move

    def reset_player(self):
        pass  

if __name__ == "__main__":

    testgame = GameState()
    game_controller = GameController(testgame)

    player1 = RandomPlayer(1)
    player2 = RandomPlayer(-1)

    winner = game_controller.start_play(player1, player2, start_player=1, is_shown=True)
    print(f"Play ends, winner is: {winner}")

    policy_value_net = PolicyValueNet(model_file=CONFIG["model_file"])
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn(testgame))  
    winner, training_data = game_controller.start_self_play(mcts_player, is_shown=False)
    print(f"Play ends, winner is: {winner}")