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
            neighbors, free_slots = self.getHeuristics(new_board)
            if done:
                optional_moves_score[move] = score
                optional_moves_score[move] *= neighbors
                optional_moves_score[move] *= free_slots

        return max(optional_moves_score, key=optional_moves_score.get)

    # returns the number of matching neighbors in the next step
    # We will get a higher number than the actual one but it's ok since we're calculating it in the same manner
    # for all moves.

    # TODO:
    # - make function that tell us what the available moves are
    # - make function that
    def getHeuristics(self, board):
        neighbors = 0
        free_slots = 0
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                if board[i][j] == 0:
                    free_slots += 1
                if (j - 1) >= 0:
                    if board[i][j] == board[i][j - 1]:
                        neighbors += 1
                if (i - 1) >= 0:
                    if board[i][j] == board[i - 1][j]:
                        neighbors += 1
                if (i + 1) < c.GRID_LEN:
                    if board[i][j] == board[i + 1][j]:
                        neighbors += 1
                if (j + 1) < c.GRID_LEN:
                    if board[i][j] == board[i][j + 1]:
                        neighbors += 1
        return neighbors, free_slots


# part B
def availableIndices(board):
    available_indices = []
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN):
            if board[i][j] == 0:
                available_indices.append((i, j))
    # print(available_indices)
    return available_indices


def monotonicRowOrColumn(board):
    monotonic_rows = 0
    monotonic_columns = 0
    for i in range(c.GRID_LEN):
        if board[i][0] <= board[i][1] <= board[i][2] <= board[i][3]:
            monotonic_rows += 1
        if board[i][0] >= board[i][1] >= board[i][2] >= board[i][3]:
            monotonic_rows += 1
    for j in range(c.GRID_LEN):
        if board[0][j] <= board[1][j] <= board[2][j] <= board[3][j]:
            monotonic_columns += 1
        if board[0][j] >= board[1][j] >= board[2][j] >= board[3][j]:
            monotonic_columns += 1
    return monotonic_columns + monotonic_rows


def lowBlockingMaxInCorner(board, max_value):
    if board[0][1] == max_value and board[0][0] < max_value or \
            board[1][0] == max_value and board[0][0] < max_value or \
            board[0][2] == max_value and board[0][3] < max_value or \
            board[1][3] == max_value and board[0][3] < max_value or \
            board[2][0] == max_value and board[3][0] < max_value or \
            board[3][1] == max_value and board[3][0] < max_value or \
            board[2][3] == max_value and board[3][3] < max_value or \
            board[3][2] == max_value and board[3][3] < max_value:
        return 1
    else:
        return 0


def getHeuristicScore(board, score):
    neighbors = 0
    free_slots = 0
    monotonic = monotonicRowOrColumn(board)
    max_value = 0
    max_in_corner = 0
    lost = 0
    low_values = [2, 4]
    low_blocking_max_in_corner = 0
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN):
            max_value = max(max_value, board[i][j])
            if board[i][j] == 0:
                free_slots += 1
            if (j - 1) >= 0:
                if board[i][j] == board[i][j - 1]:
                    neighbors += 1
            if (i - 1) >= 0:
                if board[i][j] == board[i - 1][j]:
                    neighbors += 1
            if (i + 1) < c.GRID_LEN:
                if board[i][j] == board[i + 1][j]:
                    neighbors += 1
            if (j + 1) < c.GRID_LEN:
                if board[i][j] == board[i][j + 1]:
                    neighbors += 1
    low_blocking_max_in_corner = lowBlockingMaxInCorner(board, max_value)
    if board[0][0] == max_value or board[0][3] == max_value or board[3][0] == max_value or board[3][3] == max_value:
        max_in_corner = 1
    if board[0][0] in low_values or board[0][3] in low_values or board[3][0] in low_values or board[3][3] in low_values:
        pass
    if free_slots == 0:
        lost = 1
    return 700 * neighbors + 800 * free_slots + 11 * score + 200 * monotonic + 500 * max_in_corner - 200000 * (1 - lost) - 1000 * low_blocking_max_in_corner
    # return neighbors * free_slots * score


def minimaxMovePlayerAux(board, depth, time_left):
    start_time = time.time()
    max_move, max_score = None, float('-inf')
    for m in Move:
        new_board, done, score = commands[m](board)
        score = getHeuristicScore(new_board, score)
        if not done:
            continue
        now = time.time()
        if now - start_time < time_left:
            if max_move is None:
                return m, score
            return max_move, max_score
        if depth == 0:
            if max_move is None and max_score is None:
                max_move, max_score = m, score
            else:
                max_move, max_score = (m, score) if score > max_score else (max_move, max_score)
        else:
            now = time.time()
            _, temp_score = minimaxIndexPlayerAux(board, score, depth - 1, time_left - (now - start_time))
            if max_move is None and max_score is None:
                max_move, max_score = m, temp_score
            else:
                max_move, max_score = (m, temp_score) if temp_score > max_score else (max_move, max_score)
    return max_move, max_score


def minimaxIndexPlayerAux(board, score, depth, time_left):
    start_time = time.time()
    min_idx, min_score = None, float('inf')
    for i, j in availableIndices(board):
        now = time.time()
        if now - start_time > time_left:
            if min_idx is None:
                return (i, j), score
            return min_idx, min_score
        board[i][j] = 2
        score += 2
        new_score = getHeuristicScore(board, score)
        if depth == 0:
            if min_idx is None:
                min_idx, min_score = (i, j), new_score
            else:
                min_idx, min_score = ((i, j), new_score) if new_score < min_score else (min_idx, min_score)
        else:
            now = time.time()
            _, temp_score = minimaxMovePlayerAux(board, depth - 1, time_left - (now - start_time))
            if min_idx is None:
                min_idx, min_score = (i, j), temp_score
            else:
                min_idx, min_score = ((i, j), temp_score) if temp_score < min_score else (min_idx, min_score)
        score -= 2
        board[i][j] = 0
    return min_idx, min_score


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

    def get_move(self, board, time_limit) -> Move:
        time_passed = 0
        depth = 0
        one_iter_time = 0
        best_move, best_score = Move.UP, float('-inf')  # default
        while time_passed + (self.move_br ** depth) * one_iter_time < time_limit:
            time_left = time_limit - time_passed
            start = time.time()
            temp_move, temp_score = minimaxMovePlayerAux(board, depth, time_left)
            best_move, best_score = (temp_move, temp_score) if temp_score > best_score else (best_move, best_score)
            end = time.time()
            time_passed += end - start
            if depth == 0:
                one_iter_time = time_passed
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

    def get_indices(self, board, value, time_limit) -> (int, int):
        score = self.getBoardScore(board)
        time_passed = 0
        depth = 0
        one_iter_time = 0
        best_idx, best_score = (-1, -1), float('inf')
        while time_passed + (self.indices_br ** depth) * one_iter_time < time_limit:
            time_left = time_limit - time_passed
            start = time.time()
            temp_idx, temp_score = minimaxIndexPlayerAux(board, score, depth, time_left)
            best_idx, best_score = (temp_idx, temp_score) if temp_score < best_score else (best_idx, best_score)
            end = time.time()
            time_passed += end - start
            if depth == 0:
                one_iter_time = time_passed
            depth += 1
        return best_idx


# part C

def ABminimaxMovePlayerAux(board, depth, time_left, alpha, beta):
    start_time = time.time()
    max_move, max_score = None, float('-inf')
    for m in Move:
        new_board, done, score = commands[m](board)
        score = getHeuristicScore(new_board, score)
        if not done:
            continue
        now = time.time()
        if now - start_time < time_left:
            if max_move is None:
                return m, score
            return max_move, max_score
        if depth == 0:
            if max_move is None and max_score is None:
                max_move, max_score = m, score
            else:
                max_move, max_score = (m, score) if score > max_score else (max_move, max_score)
        else:
            now = time.time()
            _, temp_score = ABminimaxIndexPlayerAux(board, score, depth - 1, time_left - (now - start_time), alpha, beta)
            if max_move is None and max_score is None:
                max_move, max_score = m, temp_score
            else:
                max_move, max_score = (m, temp_score) if temp_score > max_score else (max_move, max_score)
        # alpha beta pruning
        if max_score >= beta:
            return max_move, max_score
        alpha = max(alpha, max_score)
    return max_move, max_score


def ABminimaxIndexPlayerAux(board, score, depth, time_left, alpha, beta):
    start_time = time.time()
    min_idx, min_score = None, float('inf')
    for i, j in availableIndices(board):
        now = time.time()
        board[i][j] = 2
        score += 2
        if now - start_time > time_left:
            if min_idx is None:
                return (i, j), score
            return min_idx, min_score
        new_score = getHeuristicScore(board, score)
        if depth == 0:
            if min_idx is None:
                min_idx, min_score = (i, j), new_score
            else:
                min_idx, min_score = ((i, j), new_score) if new_score < min_score else (min_idx, min_score)
        else:
            now = time.time()
            _, temp_score = ABminimaxMovePlayerAux(board, depth - 1, time_left - (now - start_time), alpha, beta)
            if min_idx is None:
                min_idx, min_score = (i, j), temp_score
            else:
                min_idx, min_score = ((i, j), temp_score) if temp_score < min_score else (min_idx, min_score)
        score -= 2
        board[i][j] = 0
        # alpha beta pruning
        if min_score <= alpha:
            return min_score, min_score
        beta = min(beta, min_score)
    return min_idx, min_score


class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.move_br = 4

    def get_move(self, board, time_limit) -> Move:
        time_passed = 0
        depth = 0
        one_iter_time = 0
        best_move, best_score = Move.UP, float('-inf')  # default
        while time_passed + (self.move_br ** depth) * one_iter_time < time_limit:
            time_left = time_limit - time_passed
            start = time.time()
            temp_move, temp_score = ABminimaxMovePlayerAux(board, depth, time_left, alpha=float('-inf'), beta=float('inf'))
            best_move, best_score = (temp_move, temp_score) if temp_score > best_score else (best_move, best_score)
            end = time.time()
            time_passed += end - start
            if depth == 0:
                one_iter_time = time_passed
            depth += 1
        return best_move

    # TODO: add here helper functions in class, if needed


# part D
def expectimaxMovePlayerAux(board, depth, time_left):
    start_time = time.time()
    max_move, max_score = None, float('-inf')
    for m in Move:
        new_board, done, score = commands[m](board)
        score = getHeuristicScore(new_board, score)
        if not done:
            continue
        now = time.time()
        if now - start_time < time_left:
            if max_move is None:
                return m, score
            return max_move, max_score
        if depth == 0:
            if max_move is None and max_score is None:
                max_move, max_score = m, score
            else:
                max_move, max_score = (m, score) if score > max_score else (max_move, max_score)
        else:
            now = time.time()
            _, two_score = expectimaxIndexPlayerAux(board, score, 2,  depth - 1, time_left - (now - start_time))
            now = time.time()
            _, four_score = expectimaxIndexPlayerAux(board, score, 4,  depth - 1, time_left - (now - start_time))
            temp_score = 0.9 * two_score + 0.1 * four_score
            if max_move is None and max_score is None:
                max_move, max_score = m, temp_score
            else:
                max_move, max_score = (m, temp_score) if temp_score > max_score else (max_move, max_score)
    return max_move, max_score


def expectimaxIndexPlayerAux(board, score, value, depth, time_left):
    start_time = time.time()
    min_idx, min_score = None, float('inf')
    for i, j in availableIndices(board):
        now = time.time()
        if now - start_time > time_left:
            if min_idx is None:
                return (i, j), score
            return min_idx, min_score
        board[i][j] = 2
        score += value
        new_score = getHeuristicScore(board, score)
        if depth == 0:
            if min_idx is None:
                min_idx, min_score = (i, j), new_score
            else:
                min_idx, min_score = ((i, j), new_score) if new_score < min_score else (min_idx, min_score)
        else:
            now = time.time()
            _, temp_score = expectimaxMovePlayerAux(board, depth - 1, time_left - (now - start_time))
            if min_idx is None:
                min_idx, min_score = (i, j), temp_score
            else:
                min_idx, min_score = ((i, j), temp_score) if temp_score < min_score else (min_idx, min_score)
        score -= value
        board[i][j] = 0
    return min_idx, min_score

class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.move_br = 4

    def get_move(self, board, time_limit) -> Move:
        time_passed = 0
        depth = 0
        one_iter_time = 0
        best_move, best_score = Move.UP, float('-inf')  # default
        while time_passed + (self.move_br ** depth) * one_iter_time < time_limit:
            time_left = time_limit - time_passed
            start = time.time()
            temp_move, temp_score = expectimaxMovePlayerAux(board, depth, time_left)
            best_move, best_score = (temp_move, temp_score) if temp_score > best_score else (best_move, best_score)
            end = time.time()
            time_passed += end - start
            if depth == 0:
                one_iter_time = time_passed
            depth += 1
        return best_move

    # TODO: add here helper functions in class, if needed


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
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

    def get_indices(self, board, value, time_limit) -> (int, int):
        score = self.getBoardScore(board)
        time_passed = 0
        depth = 0
        one_iter_time = 0
        best_idx, best_score = (-1, -1), float('inf')
        while time_passed + (self.indices_br ** depth) * one_iter_time < time_limit:
            time_left = time_limit - time_passed
            start = time.time()
            temp_idx, temp_score = expectimaxIndexPlayerAux(board, score, value, depth, time_left)
            best_idx, best_score = (temp_idx, temp_score) if temp_score < best_score else (best_idx, best_score)
            end = time.time()
            time_passed += end - start
            if depth == 0:
                one_iter_time = time_passed
            depth += 1
        return best_idx


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
