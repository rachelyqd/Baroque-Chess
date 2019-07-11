'''Tring_Hard_BC_Player.py by Chen Bai and Chumei Yang
This is a game agent that plays Baroque chess automatically. It should follow the rules strictly.
The program should suppose the opponent as intelligent
and make its move based on the inference on the opponent’s possible move.
Our goal is for the agent to have as many captures as possible thus winning the game.
The main algorithm used is minimax search with optimization including alpha-beta pruning and Zobrist hashing

Usage Example: python BaroqueGameMaster.py Trying_Hard_BC_Player other_agent 2
'''

import BC_state_etc as BC
import time
import copy as cp
import math
import numpy as np
import random
import sys

depth_count = 0
state_count = 0
eval_count = 0
ab_count = 0

SCORES = {0: 0, 2: 200, 3: 200, 4: 200, 5: 200, 6: 300, 7: 300, 8: 500, 9: 500,
          10: 300, 11: 300, 12: 800, 13: 800, 14: 100, 15: 100}
MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
PAWN_MOVES = [(-1, 0), (0, -1), (0, 1), (1, 0)]
REGULARS = [2, 3, 4, 5, 10, 11, 14, 15]
ALL_CAPTURE = {}
remark_counter = 0
REMARKS = ["Maybe here...I don't know", "I am paying my effort", "Don't push me, I am trying", "I am thinking hard",
           "This is your move? I guess I should think harder",
           "I will make this move for this turn, and think harder next round", "Take this, I made this move by effort",
           "I paid my effort and so should you", "I tried, but I only get this", "This is the best I can do"]
DEPTH = 2
TIME = 1.0
ALPHA_BETA = 1
RULE = 1

Y_BOARD = 8
X_BOARD = 8
NUM_PIECE = 14
ZOBRIST_TABLE = np.zeros([Y_BOARD, X_BOARD, NUM_PIECE], dtype=int)
# WHITE_LIST = ["P", "L", "I", "W", "K", "C", "F"]
# BLACK_LIST = ["p", "l", "i", "w", "k", "c", "f"]
WHITE_LIST = [3, 5, 7, 9, 11, 13, 15]
BLACK_LIST = [2, 4, 6, 8, 10, 12, 14]
# PIECE_LIST = ["P", "L", "I", "W", "K", "C", "F",
#              "p", "l", "i", "w", "k", "c", "f"]
PIECE_LIST = WHITE_LIST + BLACK_LIST
VISTED_BOARD = {}


class MY_BC_STATE(BC.BC_state):
    def __hash__(self):
        return (str(self)).__hash__()

    def static_eval(self, rule):
        evalu = 0
        if rule == 0:
            evalu = self.rule_0()
        if rule == 1:
            evalu = self.rule_1()
        return evalu

    def rule_0(self):
        score = 0
        for i in range(Y_BOARD):
            for j in range(X_BOARD):
                piece = self.board[i][j]
                if piece % 2 == 1:
                    if piece == 3:
                        plus_score = 1
                    elif piece == 13:
                        plus_score = 100
                    else:
                        plus_score = 3
                    score += plus_score
                if piece % 2 == 0:
                    if piece == 2:
                        minus_score = 1
                    elif piece == 12:
                        minus_score = 100
                    else:
                        minus_score = 3
                    score -= minus_score
        return score

    def rule_1(self):
        global ALL_CAPTURE
        board = self.board
        whose_turn = self.whose_move
        ALL_CAPTURE = {}
        score = 0
        total_count = 0
        king_position = (0, 0)
        king_op_position = (0, 0)
        # traverse through the board
        for i in range(8):
            for j in range(8):
                # adding count and piece priority to total score
                # check possible new captures or new losses
                curr_piece = board[i][j]
                if curr_piece != 0:
                    if curr_piece % 2 == 1:
                        if curr_piece in REGULARS:
                            regular_move(board, i, j, 1)
                        if curr_piece == 6:
                            leaper(board, i, j, 1)
                        if curr_piece == 8:
                            imitator(board, i, j, 1)
                        if curr_piece == 12:
                            king_cap = king_capture(board, i, j, 1)
                            king_position = (i, j)
                            if king_cap:
                                for cap in king_cap:
                                    ALL_CAPTURE[cap] = king_cap[cap]
                        if ALL_CAPTURE:
                            for each_cap in ALL_CAPTURE:
                                cap_piece = ALL_CAPTURE[each_cap]
                                score += SCORES[cap_piece]
                        score += SCORES[curr_piece]
                        score += 50 + 6.25 * abs(i - 4) + 6.25 * abs(j - 4)  # center control
                        ALL_CAPTURE = {}
                    else:
                        if curr_piece in REGULARS:
                            regular_move(board, i, j, 0)
                        if curr_piece == 6:
                            leaper(board, i, j, 0)
                        if curr_piece - whose_turn + 1 == 8:
                            imitator(board, i, j, 0)
                        if curr_piece == 12:
                            king_cap = king_capture(board, i, j, 0)
                            king_op_position = (i, j)
                            if king_cap:
                                for cap in king_cap:
                                    ALL_CAPTURE[cap] = king_cap[cap]
                        if ALL_CAPTURE:
                            for each_cap in ALL_CAPTURE:
                                cap_piece = ALL_CAPTURE[each_cap]
                                score -= SCORES[cap_piece]
                        score -= SCORES[curr_piece]
                        score -= 50 + 6.25 * abs(i - 4) + 6.25 * abs(j - 4)
                        ALL_CAPTURE = {}
                    # check king position and
                    if curr_piece == 13:
                        king_position = (i, j)
                    if curr_piece == 12:
                        king_op_position = (i, j)
                    total_count += 1
        # increase the score if the king is close to the center when not many pieces left
        if total_count < 16:
            score += 200 - 25 * abs(king_position[0] - 4) - 25 * abs(king_position[1] - 4)
            score -= 200 - 25 * abs(king_op_position[0] - 4) - 25 * abs(king_op_position[1] - 4)
        ALL_CAPTURE = {}
        return score


def children_states(board, whose_turn):
    res = {}
    # check each piece for a side
    for i in range(8):
        for j in range(8):
            curr_piece = board[i][j]
            # eliminate pieces to check
            if curr_piece == 0:
                continue  # try to move empty space
            elif (curr_piece - whose_turn) % 2 != 0:
                continue  # try to move other player's piece
            elif isFrozen(board, i, j, whose_turn):
                continue
            else:
                curr_res = get_children(board, i, j, whose_turn)
                res.update(curr_res)
    return res


def get_children(board, row, col, whose_turn):
    res = []
    moves = MOVES
    curr_piece = board[row][col]
    # if the piece is a pawn, coordinator, withdrawer, freezer
    # not allowed to go over and only can land in empty space
    if curr_piece in REGULARS:
        res = regular_move(board, row, col, whose_turn)
    elif curr_piece == 12 or curr_piece == 13:
        res = king(board, row, col, whose_turn)
    elif curr_piece == 6 or curr_piece == 7:
        res = leaper(board, row, col, whose_turn)
    elif curr_piece == 8 or curr_piece == 9:
        res = imitator(board, row, col, whose_turn)
    return res


def regular_move(board, row, col, whose_turn):
    global ALL_CAPTURE
    res = {}
    curr_piece = board[row][col]
    moves = MOVES
    if curr_piece == 2 or curr_piece == 3:
        moves = PAWN_MOVES
    for move in moves:
        for multi in range(8):
            mx = move[0] * (1 + multi)
            my = move[1] * (1 + multi)
            new_row = row + mx
            new_col = col + my
            if new_row < 0 or new_row > 7 or new_col < 0 or new_col > 7:
                break
            elif board[new_row][new_col] != 0:
                break
            else:
                new_board = copy(board)  # new board to make changes
                # check captures
                capture_dict = {}
                if curr_piece == 2 or curr_piece == 3:  # pawn case
                    capture_dict = pawn_capture(board, new_row, new_col, whose_turn)
                elif curr_piece == 4 or curr_piece == 5:  # coordinator case
                    capture_dict = coordinator_capture(board, new_row, new_col, whose_turn)
                elif curr_piece == 10 or curr_piece == 11:  # withdrawer case
                    capture_dict = withdrawer_capture(board, row, col, move[0], move[1], whose_turn)
                # freezer cannot capture
                if capture_dict:
                    for cap_coord in capture_dict:
                        x, y = cap_coord
                        new_board[x][y] = 0
                        ALL_CAPTURE[cap_coord] = capture_dict[cap_coord]
                # set original pawn position to empty
                new_board[row][col] = 0
                # set new position to pawn
                new_board[new_row][new_col] = curr_piece
                new_state = MY_BC_STATE(new_board, 1 - whose_turn)
                res[new_state] = ((row, col), (new_row, new_col))
                # res.append(new_board)
    return res


def king(board, row, col, whose_turn):
    global ALL_CAPTURE
    res = {}
    possible_pos = []
    for move in MOVES:
        new_row = row + move[0]
        new_col = col + move[1]
        if new_row >= 0 and new_row < 8 and new_col >= 0 and new_col < 8:
            # if new space empty or opposite piece, can move to that position
            if board[new_row][new_col] == 0 or board[new_row][new_col] % 2 != whose_turn:
                # check if king will be captured if moved to new position
                possible_pos.append((new_row, new_col))
                for i in range(8):
                    for j in range(8):
                        if board[i][j] % 2 != whose_turn:
                            capture_dict = {}
                            if board[i][j] in REGULARS:
                                regular_move(board, i, j, 1 - whose_turn)
                                capture_dict = ALL_CAPTURE
                            elif board[i][j] == 12 + (1 - whose_turn):
                                capture_dict = king_capture(board, i, j, 1 - whose_turn)
                            elif board[i][j] == 6 + (1 - whose_turn):
                                leaper(board, i, j, 1 - whose_turn)
                                capture_dict = ALL_CAPTURE
                            for capture_coord in capture_dict:
                                if capture_coord == (new_row, new_col):
                                    if capture_dict[capture_coord] == 12 + whose_turn:
                                        possible_pos.remove((new_row, new_col))
    if possible_pos:
        for each_coor in possible_pos:
            new_board = copy(board)
            x, y = each_coor
            new_board[x][y] = 12 + whose_turn
            new_board[row][col] = 0
            new_state = MY_BC_STATE(new_board, 1 - whose_turn)
            res[new_state] = ((row, col), each_coor)
            # res.append(new_board)
    ALL_CAPTURE = {}
    return res


def leaper(board, row, col, whose_turn):
    capture_dict = {}
    res = {}
    encounter = False
    capture = (0, 0)
    for move in MOVES:
        for multi in range(8):
            mx = move[0] * (1 + multi)
            my = move[1] * (1 + multi)
            new_row = row + mx
            new_col = col + my
            # new position within board
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                new_board = copy(board)
                # if new position is not empty
                if not board[new_row][new_col] == 0:
                    # if new position is friendly piece, break
                    if board[new_row][new_col] % 2 == whose_turn:
                        break
                    # if new position opposite piece
                    elif board[new_row][new_col] % 2 != whose_turn:
                        # if already encountered an opposite piece, break
                        if encounter:
                            break
                        # else set encounter to true, set capture coordinate and continue
                        else:
                            encounter = True
                            capture = (new_row, new_col)
                            continue
                # if new position is empty
                else:
                    # if have encountered an opposite piece, perform capture
                    if encounter:
                        capture_dict[capture] = board[capture[0]][capture[1]]
                        ALL_CAPTURE[capture] = board[capture[0]][capture[1]]
                        new_board[capture[0]][capture[1]] = 0
                    # set new position to leaper
                    new_board[new_row][new_col] = 6 + whose_turn
                    new_board[row][col] = 0
                    new_state = MY_BC_STATE(new_board, 1 - whose_turn)
                    res[new_state] = ((row, col), (new_row, new_col))
                    # res.append(new_board)
        encounter = False
    return res


def imitator(board, row, col, whose_turn):
    global ALL_CAPTURE
    res = {}
    regular = True
    has_leapcap = False
    encounter = False
    for move in MOVES:
        for multi in range(8):
            mx = move[0] * (1 + multi)
            my = move[1] * (1 + multi)
            new_row = row + mx
            new_col = col + my
            if new_row < 0 or new_row > 7 or new_col < 0 or new_col > 7:
                break
            # if not empty or piece and is a friendly piece, cannot land
            elif board[new_row][new_col] != 0 and board[new_row][new_col] % 2 == whose_turn:
                break
            # if new position is opposite piece, set regular to false
            # only check leaper and king capture case
            elif not board[new_row][new_col] == 12 + 1 - whose_turn and board[new_row][new_col] % 2 != whose_turn:
                regular = False
                encounter = True
                continue
            else:
                new_board = copy(board)  # new board to make changes
                capture_dict = {}
                if regular:
                    # check captures
                    # check capture as a pawn
                    if move in PAWN_MOVES:
                        pawn_cap = pawn_capture(board, new_row, new_col, whose_turn)
                        if pawn_cap:
                            for cap_coor in pawn_cap:
                                # if capture is opposite pawn
                                if pawn_cap[cap_coor] == 2 + 1 - whose_turn:
                                    capture_dict[cap_coor] = 2 + 1 - whose_turn
                                    ALL_CAPTURE[cap_coor] = 2 + 1 - whose_turn
                    coordi_cap = coordinator_capture(board, new_row, new_col, whose_turn)
                    if coordi_cap:
                        for cap_coor in coordi_cap:
                            # if capture is opposite coordinator
                            if coordi_cap[cap_coor] == 4 + 1 - whose_turn:
                                capture_dict[cap_coor] = 4 + 1 - whose_turn
                                ALL_CAPTURE[cap_coor] = 4 + 1 - whose_turn
                    with_cap = withdrawer_capture(board, row, col, move[0], move[1], whose_turn)
                    if with_cap:
                        for cap_coor in with_cap:
                            # if capture is opposite withdrawer
                            if with_cap[cap_coor] == 10 + 1 - whose_turn:
                                capture_dict[cap_coor] = 10 + 1 - whose_turn
                                ALL_CAPTURE[cap_coor] = 10 + 1 - whose_turn
                # check if capture can be a king
                if multi == 0:
                    if board[new_row][new_col] == 12 + 1 - whose_turn:
                        ALL_CAPTURE[(new_row, new_col)] = 12 + 1 - whose_turn
                        new_board[new_row][new_col] = 8 + whose_turn
                        new_board[row][col] = 0
                        new_state = MY_BC_STATE(new_board, 1 - whose_turn)
                        res[new_state] = ((row, col), (new_col, new_col))
                        break
                leaper_cap = leaper_capture(board, row, col, new_row, new_col, move, whose_turn)
                if leaper_cap:
                    has_leapcap = True
                    for cap_coor in leaper_cap:
                        # if capture is opposite coordinator
                        if leaper_cap[cap_coor] == 6 + 1 - whose_turn:
                            capture_dict[cap_coor] = 6 + 1 - whose_turn
                            ALL_CAPTURE[cap_coor] = 6 + 1 - whose_turn
                        else:
                            has_leapcap = False
                if regular or has_leapcap:
                    # choose the best piece to capture
                    if capture_dict:
                        max_score = 0
                        nx = 0
                        ny = 0
                        for each_cap in capture_dict:
                            piece = capture_dict[each_cap]
                            piece_score = SCORES[piece]
                            if piece_score > max_score:
                                nx, ny = each_cap
                        new_board[nx][ny] = 0
                    new_board[row][col] = 0
                    new_board[new_row][new_col] = 8 + whose_turn
                    new_state = MY_BC_STATE(new_board)  # if need to update whose_move?
                    res[new_state] = ((row, col), (new_row, new_col))
                    # res.append(new_board)
                if not has_leapcap and encounter:
                    encounter = False
                    break
        regular = True
    return res


def pawn_capture(board, new_row, new_col, whose_turn):
    capture = {}
    for ad in PAWN_MOVES:
        adx, ady = ad
        if new_row + 2 * adx >= 0 and new_row + 2 * adx < 8 and new_col + 2 * ady >= 0 and new_col + 2 * ady < 8:
            # check if opponent is in between curr piece and friendly piece
            ad = board[new_row + adx][new_col + ady]
            if board[new_row + adx][new_col + ady] % 2 != whose_turn and board[new_row + adx][new_col + ady] != 0:
                if board[new_row + 2 * adx][new_col + 2 * ady] % 2 == whose_turn and board[new_row + 2 * adx][
                    new_col + 2 * ady] != 0:
                    capture[(new_row + adx, new_col + ady)] = board[new_row + adx][new_col + ady]
    return capture


def coordinator_capture(board, new_row, new_col, whose_turn):
    # check if there are any opponent pieces on current row
    capture = {}
    for i in range(8):
        # opponent on current row
        if board[new_row][i] != 0:
            if i != new_col and board[new_row][i] % 2 != whose_turn:
                # check if king is on this column
                for j in range(8):
                    if board[j][i] == 12 + whose_turn:
                        capture[(new_row, i)] = board[new_row][i]
    # check if there are any opponent pieces on current column
    for i in range(8):
        # opponent on current row
        new_piece = board[i][new_col]
        same = new_piece % 2
        if board[i][new_col] != 0:
            if i != new_row and board[i][new_col] % 2 != whose_turn:
                # check if king is on this column
                for j in range(8):
                    if board[i][j] == 12 + whose_turn:
                        capture[(i, new_col)] = board[i][new_col]
    return capture


def withdrawer_capture(board, row, col, move_x, move_y, whose_turn):
    capture = {}
    opposite_x = row - move_x
    opposite_y = col - move_y
    if opposite_x >= 0 and opposite_x < 8 and opposite_y >= 0 and opposite_y < 8:
        # check if opponent in opposite direction and is not a freezer
        if board[opposite_x][opposite_y] % 2 != whose_turn and board[opposite_x][opposite_y] % 2 != 0 \
                and board[opposite_x][opposite_y] != 14 + (1 - whose_turn):
            capture[(opposite_x, opposite_y)] = board[opposite_x][opposite_y]
    return capture


def king_capture(board, row, col, whose_turn):
    capture = {}
    for move in MOVES:
        new_row = row + move[0]
        new_col = col + move[1]
        if 0 <= new_row < 8 and 0 <= new_col < 8:
            if board[new_row][new_col] % 2 != whose_turn and board[new_row][new_col] % 2 != 0:
                capture[(new_row, new_col)] = board[new_row][new_col]
                ALL_CAPTURE[(new_row, new_col)] = board[new_row][new_col]
    return capture


def leaper_capture(board, row, col, new_row, new_col, move, whose_turn):
    capture = {}
    mx = new_row - row
    my = new_col - col
    multi = max(abs(mx), abs(my))
    move_x = round(mx / multi)
    move_y = round(my / multi)
    encounter = False
    nx = row
    ny = col
    for i in range(multi):
        nx += move_x
        ny += move_y
        # if new position is not empty
        if not board[nx][ny] == 0:
            # if new position is friendly piece, break
            if board[nx][ny] % 2 == whose_turn:
                break
            # if new position opposite piece
            elif board[nx][ny] % 2 != whose_turn:
                # if already encountered an opposite piece, break
                if encounter:
                    return {}
                # else set encounter to true, set capture coordinate and continue
                else:
                    encounter = True
                    capture[(nx, ny)] = board[nx][ny]
    return capture


def isFrozen(board, row, col, whose_turn):
    for move in MOVES:
        mx, my = move
        curr_x = row + mx
        curr_y = col + my
        if curr_x < 0 or curr_x > 7 or curr_y < 0 or curr_y > 7:
            continue
        if board[curr_x][curr_y] - (1 - whose_turn) == 14:
            return True
    return False


def copy(board):
    return [r[:] for r in board]


def to_string(board):
    s = ''
    for r in range(8):
        for c in range(8):
            s += BC.CODE_TO_INIT[board[r][c]] + " "
        s += "\n"
    return s


def searcher(current, depth, time_limit, alpha_beta):
    time_limit = time.time() + time_limit * 0.9
    who = current.whose_move
    new_who = 1 - who
    current.__class__ = MY_BC_STATE
    best_eval, move, new_state = minimax_a_b_search(current, -math.inf, math.inf, depth, 0, who, time_limit, alpha_beta)
    new_state.whose_move = new_who
    return best_eval, new_state, move


def minimax_a_b_search(current, a, b, depth, d, white, time_limit, a_b):
    global depth_count, state_count, eval_count, ab_count
    new_move = None
    re_state = current
    depth_count = max(depth_count, d)  # update depth count
    children = children_states(current.board, current.whose_move)
    if depth == d or len(children) == 0:  # base case
        eval_count += 1
        current.__class__ = MY_BC_STATE
        return current.static_eval(RULE), new_move, re_state
    state_count += 1  # update state count
    if white:  # maximizer
        max_eval = -math.inf
        for child in children:
            if time.time() >= time_limit:
                if new_move is None:
                    new_move = children[child]
                break
            child_hash = board_hash_value(child.board)  # find hash value for the board
            # normal computation
            if child_hash not in VISTED_BOARD:
                evalu, _, _ = minimax_a_b_search(child, a, b, depth, d + 1, False, time_limit, a_b)
                VISTED_BOARD[child_hash] = evalu
            else:  # zobrist
                evalu = VISTED_BOARD[child_hash]
            if evalu > max_eval:  # update evaluation and returned state
                max_eval = evalu
                new_move = children[child]
                re_state = child
                if time.time() >= time_limit:
                    break
            if a_b:
                a = max(a, evalu)  # update alpha
                if a >= b:
                    ab_count += 1  # update a_b_cut counter
                    break
        return max_eval, new_move, re_state
    else:  # minimizer
        min_eval = math.inf
        for child in children:
            if time.time() >= time_limit:
                if new_move is None:
                    new_move = children[child]
                    re_state = child
                break
            child_hash = board_hash_value(child.board)
            evalu = 0
            if child_hash not in VISTED_BOARD:
                evalu, _, _ = minimax_a_b_search(child, a, b, depth, d + 1, True, time_limit, a_b)
            else:
                evalu = VISTED_BOARD[child_hash]
            if evalu < min_eval:  # update evaluation and returned state
                min_eval = evalu
                new_move = children[child]
                re_state = child
                if time.time() >= time_limit:
                    break
            if a_b:
                b = min(b, evalu)
                if a >= b:
                    ab_count += 1  # update a_b_cut counter
                    break
        return min_eval, new_move, re_state


def makeMove(currentState, currentRemark, timelimit=10):
    global remark_counter
    best_eval, newState, move = searcher(currentState, DEPTH, timelimit, ALPHA_BETA)
    if remark_counter < 10:
        newRemark = REMARKS[remark_counter]
        remark_counter += 1
    else:
        remark_counter = 0
        newRemark = REMARKS[remark_counter]
        remark_counter += 1
    return [[move, newState], newRemark]


def nickname():
    return "Trying_Hard"


def introduce():
    return "I'm Trying_Hard, a newbie Baroque Chess agent by Chen Bai and Chumei Yang. " \
           "I will try hard to make any move"


def prepare(player2Nickname):
    make_hash_table()
    return 'Let us play!'


def piece_index(piece):
    for i in range(len(PIECE_LIST)):
        if piece == PIECE_LIST[i]:
            return i


# make a hash table that indicate the hash value for each piece laying on each grid
def make_hash_table():
    global ZOBRIST_TABLE
    for i in range(Y_BOARD):
        for j in range(X_BOARD):
            for k in range(NUM_PIECE):
                ZOBRIST_TABLE[i][j][k] = random.randint(0, 4294967296)  # might be changed "sys.maxsize"


# calculate the hash value for a given board
def board_hash_value(board):
    total_hash = 0
    for i in range(Y_BOARD):
        for j in range(X_BOARD):
            piece = board[i][j]
            if piece != 0:
                hash_piece = ZOBRIST_TABLE[i][j][piece_index(piece)]
                total_hash ^= hash_piece
    return total_hash

