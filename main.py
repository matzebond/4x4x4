#!env python3
import numpy as np
import numpy.random

from functools import reduce
# import itertools as ito

rng = numpy.random.default_rng()

side_length = 4
game_dimensions = 3
input_dimensions = game_dimensions - 1
slots = side_length**game_dimensions
full_positions = reduce(np.multiply, np.arange(slots, slots // 2, -1))
# full_positions = 2_515_100_142_361_640_960 # for 4,3d

board_shape = [side_length] * game_dimensions

INIT_BOARD = np.zeros(board_shape, dtype=np.uint8)


def random_board():
    return rng.choice(3, board_shape)


# def random_full_pos():
#     pass


def game(p1_move_func,
         p2_move_func,
         init_board=None,
         turn_p1=True,
         max_turns=99,
         viz=None):
    board = INIT_BOARD.copy() if init_board is None else init_board
    turn = 0
    moves = []
    while not (winner := check_winner(board)):
        print(f"{turn=}")
        player = 1 if turn_p1 else 2
        move_func = p1_move_func if turn_p1 else p2_move_func
        move = move_func(board, 1)
        moves.append(move)
        print(f"p{1 if turn_p1 else 2}: {move=}")
        make_move(board, 1 if turn_p1 else 2, move)
        if viz is not None:
            viz(board)
        turn_p1 = not turn_p1
        turn += 1
        if turn >= max_turns:
            return board, moves

    winner_func = p1_move_func if winner == 1 else p2_move_func
    print("And the winner is player", winner,
          f"playing strategy \"{winner_func.__name__}\"")
    return board, moves
    # raise Exception(f"max {turn=}reached")


def manual_move(board=None, player=None, tries=3):
    while tries > 0:
        try:
            i = input().split(',')
            move = np.array(list(map(int, i)))
            if move in legal_moves(board):
                return move
            else:
                tries -= 1
                print("Illegal move!")
        except:
            tries -= 1
            print("try again")


def random_move(board=None, player=None):
    return rng.integers([side_length] * input_dimensions)


def legal_moves(board):
    legal_mask = board[-1] == 0
    moves = np.transpose(legal_mask.nonzero())
    return moves


def iterate_legal_positions(board, turn_p1=True):
    for move in legal_moves(board):
        make_move(board, turn_p1, move)
        yield board
        revert_move(board, move)


def random_legal_move(board, player=None):
    moves = legal_moves(board)
    assert moves.size >= 0
    return rng.choice(moves)


def cpu_move(board, player=None, algo=minmax, depth=2):
    _, move = algo(board, depth=depth, turn_p1=player == 1)
    return move if move is not None else rng.choice(legal_moves(board))


def cpu_move_rand(board, player=None, algo=alphabeta, depth=2):
    _, moves = algo(board, depth=depth, turn_p1=player == 1)
    return rng.choice(moves if len(moves) > 0 else legal_moves(board))


def make_move(board, player, move):
    stick = move_to_stick(board, move)
    free = free_on_stick(stick)
    if free is None:
        raise Exception("impossible position")
    stick[free] = player
    return board


def make_moves(board, turn_p1, moves):
    for move in moves:
        player = 1 if turn_p1 else 2
        make_move(board, player, move)
        turn_p1 = not turn_p1
    return board


def revert_move(board, move):
    stick = move_to_stick(board, move)
    last = last_on_stick(stick)
    stick[last] = 0
    return board


def move_to_stick(board, move):
    # TODO fixed dimension
    return board[:, move[0], move[1]]


def free_on_stick(stick):
    for i, a in enumerate(stick):
        if a == 0:
            return i
    return None


def last_on_stick(stick):
    for i, a in enumerate(stick[::-1]):
        if a != 0:
            return len(stick) - 1 - i
    return -1


def sticks(board):
    for i in range(side_length):
        for j in range(side_length):
            for k in range(game_dimensions):
                for l in range(k + 1, game_dimensions):
                    indices = [slice(0, None)] * game_dimensions
                    indices[k] = i
                    indices[l] = j
                    indices = tuple(indices)
                    yield board[indices]
        for k in range(game_dimensions):
            indices = [slice(0, None)] * game_dimensions
            indices[k] = i
            indices = tuple(indices)
            yield board[indices].diagonal()
            yield np.flipud(board[indices]).diagonal()


def reduce_stick(line):
    res = 3
    for a in line:
        res = np.bitwise_and(res, a)
    return res


# TODO return highest to build to free position, hard
# TODO return highest free position
def eval_stick(stick):
    player = None
    free = 0
    for x in stick:
        if x == 0:
            free += 1
        elif player is None:
            player = x
        elif x != player:
            return None, 0
    return player, free


def heuristic(board, turn_p1=True) -> float:
    threads = np.zeros((2, side_length), np.uint8)
    for stick in sticks(board):
        player, free = eval_stick(stick)
        if player is not None:
            if free == 0:
                return -((player - 1) * 2 - 1)
            threads[player - 1, free] += 1

    t1 = max(0.25, float(threads[0][1]))
    t2 = max(0.25, float(threads[1][1]))
    if turn_p1:
        t2 *= 0.8
    else:
        t1 *= 0.8
    return (t1 - t2) / (t1 + t2)


def check_winner(board):
    val = heuristic(board, True)
    if val == 1:
        return 1
    elif val == -1:
        return 2
    else:
        return 0


def minmax(board, depth=1, turn_p1=True):
    val = heuristic(board, turn_p1)
    if val == 1 or val == -1:
        return val, None
    if depth <= 0:
        return val, None

    best = -1 if turn_p1 else 1
    best_move = None
    for move in legal_moves(board):
        make_move(board, 1 if turn_p1 else 2, move)
        score, score_move = minmax(board, depth - 1, not turn_p1)
        if (turn_p1 and score > best) \
           or (not turn_p1 and score < best):
            best = score
            best_move = move
        revert_move(board, move)

    return best, best_move


def alphabeta(board, depth, turn_p1=True, alpha=-1, beta=1):
    val = heuristic(board, turn_p1)
    if val == 1 or val == -1:
        return val, []
    if depth <= 0:
        return val, []

    best_move = []
    if turn_p1:
        value = -1
        for board in iterate_legal_positions(board, not turn_p1):
            score, score_move = alphabeta(board, depth - 1, not turn_p1, alpha,
                                          beta)
            if score >= value:
                value = score
                best_move.append(move)
            if value >= beta:
                break
            alpha = max(alpha, value)
        return value, best_move
    else:
        value = +1
        for board in iterate_legal_positions(board, not turn_p1):
            score, score_move = alphabeta(board, depth - 1, not turn_p1, alpha,
                                          beta)
            if score <= value:
                value = score
                best_move.append(move)
            if value <= alpha:
                break
            beta = min(beta, value)
        return value, best_move


def printb(board):

    def cell(c):
        if c == 0:
            return "∙"
        elif c == 1:
            return "■"
        elif c == 2:
            return "□"
        return c

    for z in reversed(range(side_length)):
        for y in reversed(range(side_length)):
            print(" " * y, end="")
            for x in range(side_length):
                c = board[z, y, x]
                f = free_on_stick(board[:, y, x])
                if z == f:
                    print("∘", end=" ")
                else:
                    print(cell(c), end=" ")
            print()
        print()


if __name__ == "__main__":
    game(manual_move, cpu_move, viz=printb)
