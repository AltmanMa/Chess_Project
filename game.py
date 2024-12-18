import numpy as np
from collections import deque
from config import CONFIG

def all_moves():
        """
        Generate all possible legal moves for the chess game, including all legal moves
        for all chess pieces and fixed moves for specific pieces like advisors and bishops.
        """
        index2move = {}
        move2index = {}

        # Define row and column indices
        rows = range(10)  # 0 to 9
        cols = range(9)   # 0 to 8

        # Predefined legal moves for advisors and bishops
        gmoves = [
            (0, 3, 1, 4), (1, 4, 0, 3), (0, 5, 1, 4), (1, 4, 0, 5),
            (2, 3, 1, 4), (1, 4, 2, 3), (2, 5, 1, 4), (1, 4, 2, 5),
            (9, 3, 8, 4), (8, 4, 9, 3), (9, 5, 8, 4), (8, 4, 9, 5),
            (7, 3, 8, 4), (8, 4, 7, 3), (7, 5, 8, 4), (8, 4, 7, 5)
        ]
        mmoves = [
        # Red 方Minister
            (0, 2, 2, 0), (2, 0, 0, 2), (0, 2, 2, 4), (2, 4, 0, 2),
            (2, 0, 4, 2), (4, 2, 2, 0), (4, 2, 2, 4), (2, 4, 4, 2),
            (2, 4, 0, 6), (0, 6, 2, 4), (2, 4, 4, 6), (4, 6, 2, 4),
            (0, 6, 2, 8), (2, 8, 0, 6), (2, 8, 4, 6), (4, 6, 2, 8),
        # Black 方Minister
            (7, 0, 9, 2), (9, 2, 7, 0), (7, 4, 9, 6), (9, 6, 7, 4),
            (7, 0, 5, 2), (5, 2, 7, 0), (7, 4, 5, 6), (5, 6, 7, 4),
            (9, 2, 7, 4), (7, 4, 9, 2), (5, 6, 7, 8), (7, 8, 5, 6),
            (5, 2, 7, 4), (7, 4, 5, 2), (9, 6, 7, 8), (7, 8, 9, 6)
        ]

        # Helper function to convert (start_row, start_col, end_row, end_col) to action string
        def move_to_action(start, end):
            return f"{start[0]}{start[1]}{end[0]}{end[1]}"

        # Generate legal moves for general pieces
        move_idx = 0
        for start_row in rows:
            for start_col in cols:
            # Possible destinations
                horizontal_moves = [(start_row, c) for c in cols]  # Same row
                vertical_moves = [(r, start_col) for r in rows]    # Same column
                knight_moves = [
                    (start_row + dr, start_col + dc)
                    for dr, dc in [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]
                    if 0 <= start_row + dr < 10 and 0 <= start_col + dc < 9
                ]

                all_moves = set(horizontal_moves + vertical_moves + knight_moves)
                for end_pos in all_moves:
                    if (start_row, start_col) != end_pos:  # Avoid no-op moves
                        action = move_to_action((start_row, start_col), end_pos)
                        index2move[move_idx] = action
                        move2index[action] = move_idx
                        move_idx += 1

        # Add predefined moves for advisors
        for move in gmoves:
            action = move_to_action((move[0], move[1]), (move[2], move[3]))
            index2move[move_idx] = action
            move2index[action] = move_idx
            move_idx += 1

    # Add predefined moves for bishops
        for move in mmoves:
            action = move_to_action((move[0], move[1]), (move[2], move[3]))
            index2move[move_idx] = action
            move2index[action] = move_idx
            move_idx += 1

        return index2move, move2index


index2move, move2index = all_moves()

class GameState:
    def __init__(self):
        # initialize the borad: 0 repersent empty, positive for Red, negative for black
        self.current_state = np.zeros((10, 9), dtype=int)
        self.current_player = 1  # 1: Red, -1: black
        self.state_deque = deque(maxlen=4)  # measure the history states
        self.round_count = 0
        self.last_move = None
        self.round_limit = CONFIG['round_limit'] if CONFIG['round_limit'] else 200
        self.init_board()
        if CONFIG['use_customized_game']:
            state = np.array(CONFIG['initial_board'])
            player = CONFIG['initial_player']
            self.set_board(state, player)


    def init_board(self):
        """
        Initialize the state of the board
        """
        self.current_state[0] = [-1, -2, -3, -4, -5, -4, -3, -2, -1]
        self.current_state[9] = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        self.current_state[2][1] = self.current_state[2][7] = -6
        self.current_state[7][1] = self.current_state[7][7] = 6
        self.current_state[3][0::2] = -7
        self.current_state[6][0::2] = 7
        self.state_deque.clear()
        self.state_deque.append(np.copy(self.current_state))

    def get_training_state(self):
        """
        Transfer the current borad stataus (10, 9) for training (9, 10, 9), Using one-hot
        """
        arraymap = {
            1: np.array([1, 0, 0, 0, 0, 0, 0]),    # Red Chariot
            2: np.array([0, 1, 0, 0, 0, 0, 0]),    # Red Knight
            3: np.array([0, 0, 1, 0, 0, 0, 0]),    # Red Minister
            4: np.array([0, 0, 0, 1, 0, 0, 0]),    # Red Guradian
            5: np.array([0, 0, 0, 0, 1, 0, 0]),    # Red General
            6: np.array([0, 0, 0, 0, 0, 1, 0]),    # Red Cannon
            7: np.array([0, 0, 0, 0, 0, 0, 1]),    # Red Warrior
            -1: np.array([-1, 0, 0, 0, 0, 0, 0]),  # Black Chariot
            -2: np.array([0, -1, 0, 0, 0, 0, 0]),  # Black Knight
            -3: np.array([0, 0, -1, 0, 0, 0, 0]),  # Black Minister
            -4: np.array([0, 0, 0, -1, 0, 0, 0]),  # Black Guradian
            -5: np.array([0, 0, 0, 0, -1, 0, 0]),  # Black General
            -6: np.array([0, 0, 0, 0, 0, -1, 0]),  # Black Cannon
            -7: np.array([0, 0, 0, 0, 0, 0, -1]),  # Black Warrior
            0: np.array([0, 0, 0, 0, 0, 0, 0]),    # empty
        }

        training_state = np.zeros((9, 10, 9), dtype=np.float32)

        # on 0~6 Level, show the type of the piece
        for i in range(10):
            for j in range(9):
                piece = self.current_state[i][j]
                training_state[:7, i, j] = arraymap[piece]

        # fill in the most recent move
        if self.last_move:
            move = self.last_move
            start_row, start_col = int(move[0]), int(move[1])
            end_row, end_col = int(move[2]), int(move[3])
            training_state[7, start_row, start_col] = -1  
            training_state[7, end_row, end_col] = 1

        # Fill in the gamer id
        if self.current_player == 1:  # Red
            training_state[8, :, :] = 1
        else:  # Black
            training_state[8, :, :] = -1

        return training_state

    

    def is_valid_move(self, start, end):
        """
        Primarily Check a move is valid
        """
        sr, sc = start
        er, ec = end
        if not (0 <= sr < 10 and 0 <= sc < 9 and 0 <= er < 10 and 0 <= ec < 9):
            return False  # Check whether outside the board
        piece = self.current_state[sr][sc]
        target = self.current_state[er][ec]
        if piece * self.current_player <= 0:
            return False  # Start from place without current player's piece
        if target * self.current_player > 0:
            return False  # End at a place with current player's piece
        # to fill in

        return True

    def get_all_legal_moves(self):
        #get all legal moves at the current state
        legal_moves = []
        for y in range(10):
            for x in range(9):
                piece = self.current_state[y][x]
                if piece * self.current_player > 0:  # the piece belongs to current player
                    targets = self.generate_possible_targets(piece, (y, x))
                    for target in targets:
                        if self.is_valid_move((y, x), target):
                            move = f"{y}{x}{target[0]}{target[1]}"
                            simulated_board = self.simulate_move((y, x), target)
                            if not self.simulate_is_face_to_face(simulated_board):
                                if len(self.state_deque) < 4 or not np.array_equal(
                                    simulated_board, self.state_deque[-4]
                                ):
                                    legal_moves.append(move)
        if not legal_moves:
            print(f"No legal moves generated. Current Player is: {self.current_player} and Round numer is {self.round_count}")
        return legal_moves


    def generate_possible_targets(self, piece, position):
        """
        Generate all possible targets based on differnt piece types
        """
        moves = []
        y, x = position
        if abs(piece) == 1:  # Chariot
            # move in row
            for i in range(x - 1, -1, -1):
                moves.append((y, i))
                if self.current_state[y][i] != 0:
                    break
            for i in range(x + 1, 9):
                moves.append((y, i))
                if self.current_state[y][i] != 0:
                    break
            # move in column
            for i in range(y - 1, -1, -1):
                moves.append((i, x))
                if self.current_state[i][x] != 0:
                    break
            for i in range(y + 1, 10):
                moves.append((i, x))
                if self.current_state[i][x] != 0:
                    break
        elif abs(piece) == 2:  # Knight
            for dy, dx in [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < 10 and 0 <= nx < 9:
                    # check whether the Knight is blocked
                    if abs(dy) == 2 and self.current_state[y + dy // 2][x] == 0:
                        moves.append((ny, nx))
                    elif abs(dx) == 2 and self.current_state[y][x + dx // 2] == 0:
                        moves.append((ny, nx))

        elif abs(piece) == 3:  # Minister
            for dy, dx in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < 10 and 0 <= nx < 9:
                    if self.current_state[y + dy // 2][x + dx // 2] == 0:  # check whether the minister is blocked
                        if (piece < 0 and ny <= 4) or (piece > 0 and ny >= 5):  # minister cannot get through the river
                            moves.append((ny, nx))

        elif abs(piece) == 4:  # Guardian
            for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < 10 and 0 <= nx < 9:
                    if 3 <= nx <= 5 and ((piece < 0 and ny <= 2) or (piece > 0 and ny >= 7)):
                        moves.append((ny, nx))

        elif abs(piece) == 5:  # General
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < 10 and 0 <= nx < 9:
                    if 3 <= nx <= 5 and ((piece < 0 and ny <= 2) or (piece > 0 and ny >= 7)):
                        moves.append((ny, nx))

        elif abs(piece) == 6:  # Cannon
            # in row direction
            hits = False
            for i in range(x - 1, -1, -1):
                if self.current_state[y][i] != 0:
                    if not hits:
                        hits = True
                    else:
                        moves.append((y, i))
                        break
                elif not hits:
                    moves.append((y, i))
            hits = False
            for i in range(x + 1, 9):
                if self.current_state[y][i] != 0:
                    if not hits:
                        hits = True
                    else:
                        moves.append((y, i))
                        break
                elif not hits:
                    moves.append((y, i))
            # in column direction
            hits = False
            for i in range(y - 1, -1, -1):
                if self.current_state[i][x] != 0:
                    if not hits:
                        hits = True
                    else:
                        moves.append((i, x))
                        break
                elif not hits:
                    moves.append((i, x))
            hits = False
            for i in range(y + 1, 10):
                if self.current_state[i][x] != 0:
                    if not hits:
                        hits = True
                    else:
                        moves.append((i, x))
                        break
                elif not hits:
                    moves.append((i, x))

        elif abs(piece) == 7:  # warrior
            if piece < 0:  # Black warrior
                if y + 1 < 10:
                    moves.append((y + 1, x))
                if y >= 5:
                    if x - 1 >= 0:
                        moves.append((y, x - 1))
                    if x + 1 < 9:
                        moves.append((y, x + 1))
            else:  # Red warrior
                if y - 1 >= 0:
                    moves.append((y - 1, x))
                if y <= 4:
                    if x - 1 >= 0:
                        moves.append((y, x - 1))
                    if x + 1 < 9:
                        moves.append((y, x + 1))
        return moves
    def simulate_move(self, start, end):
        """
        Similuate the state after current move
        :param start: Start place (row, col)
        :param end: End Place (row, col)
        :return: state
        """
        sr, sc = start
        er, ec = end
        simulated_board = np.copy(self.current_state)
        simulated_board[er][ec] = simulated_board[sr][sc]
        simulated_board[sr][sc] = 0
        return simulated_board

    def simulate_is_face_to_face(self, board):
        """
        Check whether face to face
        :return: Boolean
        """
        k_x, k_y = None, None
        K_x, K_y = None, None

        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece == 5:  # red general
                    K_x, K_y = x, y
                elif piece == -5:  # black general
                    k_x, k_y = x, y
        if k_x == K_x:  
            for i in range(min(k_y, K_y) + 1, max(k_y, K_y)):
                if board[i][k_x] != 0:
                    return False
            return True
        return False
    def is_face_to_face(self):
        """
        Check whether the generals are face to face
        """
        k_x, k_y = None, None
        K_x, K_y = None, None

        for y in range(10):
            for x in range(9):
                piece = self.current_state[y][x]
                if piece == 5:  # 
                    K_x, K_y = x, y
                elif piece == -5:  # 
                    k_x, k_y = x, y

        if k_x == K_x:  # in the same column
            for i in range(min(k_y, K_y) + 1, max(k_y, K_y)):
                if self.current_state[i][k_x] != 0:
                    return False
            return True
        return False
    def make_move(self, start, end=None):
        """
        Execute a move on the board.

        :param start: Can be either a tuple (start_row, start_col) or a string "6050" (action string).
        :param end: Optional if `start` is a string; otherwise, (end_row, end_col).
        """
        if isinstance(start, str):
        # If `start` is a string, parse it into (start_row, start_col) and (end_row, end_col)
            start_row, start_col, end_row, end_col = int(start[0]), int(start[1]), int(start[2]), int(start[3])
            start = (start_row, start_col)
            end = (end_row, end_col)

        if end is None:
            raise ValueError("End position must be specified when using tuple input")

        if not self.is_valid_move(start, end):
            raise ValueError("Illegal Move")

        sr, sc = start
        er, ec = end

        # Execute the move
        self.current_state[er][ec] = self.current_state[sr][sc]
        self.current_state[sr][sc] = 0

        # Update the last move
        self.last_move = (sr, sc, er, ec)  # Store the move as (start_row, start_col, end_row, end_col)

        # Update the historical state
        self.state_deque.append(np.copy(self.current_state))

        # Switch the current player
        self.current_player *= -1

        # Increment the round count
        self.round_count += 1

    def is_game_over(self):
        """
        check whether the game is over
        :return: Boolean whether over, red win: 1, black win: -1
        """
        if not np.any(self.current_state == 5):  # Red Genearl get killed
            return True, -1
        if not np.any(self.current_state == -5):  # Black Genearl get killed
            return True, 1
        if self.round_count >= self.round_limit:  # extend the limit
            return True, 0
        return False, None

    def render(self):
        """
        portrait the state of the board
        """
        for row in self.current_state:
            print(' '.join(f'{x:2}' for x in row))
        print()

    def set_board(self, new_state, current_player=1, round_count=0, last_move=None):
        """
        Set the Board State Mannally, only for adjusting
        
        :param new_state: (10, 9), borad
        :param current_player: 1 for Red, -1 for Black
        :param round_count: Current Round Number
        :param last_move: Last Move,  (start_row, start_col, end_row, end_col)
        """
        if new_state.shape != (10, 9):
            raise ValueError("Board Shape Error: please make sure (10, 9)")
        self.current_state = new_state
        self.current_player = current_player
        self.round_count = round_count
        self.last_move = last_move
        self.state_deque.clear()
        self.state_deque.append(np.copy(new_state))
        print("Board state updated successfully.")

def flip_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(string[index]))
        else:
            new_str += (str(8 - int(string[index])))
    return new_str

if __name__ == "__main__":
    # Test
    game = GameState()
    new_state = np.array([[-1,  0, -3,  0,  6, -4,  0,  0, -1],
                        [ 0,  0,  0,  0, -4, -2,  0,  0,  0],
                        [ 0,  0, -6,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0, -7,  0,  0],
                        [-7,  0, -7,  0, -7,  0, -3,  0, -7],
                        [ 0, -6,  0,  6,  0,  0,  0,  0,  0],
                        [ 7,  0,  7,  0,  7,  0,  7,  0,  7],
                        [ 0,  0,  2,  0,  0,  0,  0,  0,  3],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  2],
                        [ 1,  0,  3,  4,  5,  4,  0,  0,  1],
                        ])
    # New_state = np.zeros((10, 9), dtype=int)
    # New_state[0] = [1, 0, 0, 0, -5, 0, 0, 0, 0]
    # New_state[9] = [0, 0, 0, 0, 5, 0, 0, 0, 0]
    # New_state[7][4] = 3
    # New_state[1][1] = 1
    game.set_board(new_state, -1)
    game.render()
    is_over, winner = game.is_game_over()
    print(f"game is over: {is_over}")
    print(f"winner is : {winner}")
    
    legal_moves = game.get_all_legal_moves()
    print("Legal move:", legal_moves)
    #print(game.get_training_state().shape)
    # Move
    try:
        move = legal_moves[0]  # first legal move
        start = (int(move[0]), int(move[1]))
        end = (int(move[2]), int(move[3]))
        game.make_move(start, end)
        game.render()
    except ValueError as e:
        print(e)
        
    is_over, winner = game.is_game_over()
    print(f"game is over: {is_over}")
    print(f"winner is : {winner}")

    #test draw
    """ for i in range(31):
        if game.is_game_over()[0]:
            print("Game End:", "Draw" if game.is_game_over()[1] == 0 else f"Winner: {game.is_game_over()[1]}")
            break
        moves = game.get_all_legal_moves()
        if moves:
            move = moves[0]
            start = (int(move[0]), int(move[1]))
            end = (int(move[2]), int(move[3]))
            game.make_move(start, end) """
