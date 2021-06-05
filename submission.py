import constants as c
import logic
import random
from AbstractPlayers import *
import time

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score

        return max(optional_moves_score, key=optional_moves_score.get)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """
    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, c.GRID_LEN - 1)
        b = random.randint(0, c.GRID_LEN - 1)
        while board[a][b] != 0:
            a = random.randint(0, c.GRID_LEN - 1)
            b = random.randint(0, c.GRID_LEN - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

        # TODO: add here if needed

    # We'll choose by multiplying the heuristic values of all the heuristic functions
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score
                optional_moves_score[move] *= self.getFreeSlotsNum(new_board)
                optional_moves_score[move] *= self.getMatchingNeighbors(new_board)

        return max(optional_moves_score, key=optional_moves_score.get)


    # returns the number of free slots in the board
    def getFreeSlotsNum(self, board):
        free_slots = 0
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                if board[i][j] == 0:
                    free_slots += 1
        return free_slots

    # returns the number of matching neighbors in the next step
    # We will get a higher number than the actual one but it's ok since we're calculating it in the same manner
    # for all moves.
    def getMatchingNeighbors(self, board):
        neighbors = 0
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                if (j-1) >= 0:
                    if board[i][j] == board[i][j-1]:
                        neighbors += 1
                if (i-1) >= 0:
                    if board[i][j] == board[i-1][j]:
                        neighbors += 1
                if (i+1) < c.GRID_LEN:
                    if board[i][j] == board[i+1][j]:
                        neighbors += 1
                if (j+1) < c.GRID_LEN:
                    if board[i][j] == board[i][j+1]:
                        neighbors += 1
        return neighbors




# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.board = None
        self.score = 0
        self.move_br = 4
        # TODO: add here if needed

    def availableIndices(self, board):
        available_indices = []
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                if board[i][j] == 0:
                    available_indices.append((i, j))
        return available_indices

    def minimaxMovePlayerAux(self, board, depth):
        max_move, max_score = None, None
        for m in Move:
            new_board, done, score = commands[m](board)
            if not done:
                continue
            if depth == 0:
                if max_move is None and max_score is None:
                    max_move, max_score = m, score
                else:
                    max_move, max_score = (m, score) if score > max_score else (max_move, max_score)
            else:
                _, temp_score = self.minimaxIndexPlayerAux(board, score, depth - 1)
                if max_move is None and max_score is None:
                    max_move, max_score = m, temp_score
                else:
                    max_move, max_score = (m, temp_score) if temp_score > max_score else (max_move, max_score)
        return max_move, max_score

    def minimaxIndexPlayerAux(self, board, score, depth):
        min_idx, min_score = None, None
        for i, j in self.availableIndices(board):
            board[i][j] = 2
            score += 2
            if depth == 0:
                if min_idx is None and min_score is None:
                    min_idx, min_score = (i, j), score
                else:
                    min_idx, min_score = ((i, j), score) if score < min_score else (min_idx, min_score)
            else:
                _, temp_score = self.minimaxMovePlayerAux(board, depth - 1)
                if min_idx is None and min_score is None:
                    min_idx, min_score = (i, j), temp_score
                else:
                    min_idx, min_score = ((i, j), temp_score) if temp_score < min_score else (min_idx, min_score)
            score += 2
            board[i][j] = 0
        return min_idx, min_score

    def get_move(self, board, time_limit) -> Move:
        time_passed = 0
        depth = 0
        best_move = Move.UP # default
        while time_passed * self.move_br < time_limit:
            start = time.time()
            best_move, _ = self.minimaxMovePlayerAux(board, depth)
            end = time.time()
            time_passed += end - start
            depth += 1
        return best_move







class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        self.indices_br = 4 * 4

    def getBoardScore(self, board):
        score = 0
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                score += board[i][j]
        return score

    def availableIndices(self, board):
        available_indices = []
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                if board[i][j] == 0:
                    available_indices.append((i, j))
        return available_indices

    def minimaxMovePlayerAux(self, board, depth):
        max_move, max_score = None, None
        for m in Move:
            new_board, done, score = commands[m](board)
            if not done:
                continue
            if depth == 0:
                if max_move is None and max_score is None:
                    max_move, max_score = m, score
                else:
                    max_move, max_score = (m, score) if score > max_score else (max_move, max_score)
            else:
                _, temp_score = self.minimaxIndexPlayerAux(board, score, depth - 1)
                if max_move is None and max_score is None:
                    max_move, max_score = m, temp_score
                else:
                    max_move, max_score = (m, temp_score) if temp_score > max_score else (max_move, max_score)
        return max_move, max_score

    def minimaxIndexPlayerAux(self, board, score, depth):
        min_idx, min_score = None, None
        for i, j in self.availableIndices(board):
            board[i][j] = 2
            score += 2
            if depth == 0:
                if min_idx is None and min_score is None:
                    min_idx, min_score = (i, j), score
                else:
                    min_idx, min_score = ((i, j), score) if score < min_score else (min_idx, min_score)
            else:
                _, temp_score = self.minimaxMovePlayerAux(board, depth - 1)
                if min_idx is None and min_score is None:
                    min_idx, min_score = (i, j), temp_score
                else:
                    min_idx, min_score = ((i, j), temp_score) if temp_score < min_score else (min_idx, min_score)
            score += 2
            board[i][j] = 0
        return min_idx, min_score

    def get_indices(self, board, value, time_limit) -> (int, int):
        score = self.getBoardScore(board)
        time_passed = 0
        depth = 0
        best_idx = None
        while time_passed * self.indices_br < time_limit:
            start = time.time()
            best_idx, _ = self.minimaxIndexPlayerAux(board, score, depth)
            end = time.time()
            time_passed += end - start
            depth += 1
        return best_idx


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed

