import core
import utils

from chess.moves import Castling, Move
from chess.board import BlindBoard

CASTLING_DIFF = {
    Castling.Side.KING :
        [ BlindBoard.Diff({'e1', 'h1'}, {'f1','g1'}, set()),
          BlindBoard.Diff({'e8', 'h8'}, {'f8','g8'}, set()) ],

    Castling.Side.QUEEN :
        [ BlindBoard.Diff({'e1', 'a1'}, {'c1','d1'}, set()),
          BlindBoard.Diff({'e8', 'a8'}, {'c8','d8'}, set()) ],
}

CASTLING_MOVE_LENGTH = [2, 2, 0]
TAKE_MOVE_LENGTH     = [1, 0, 1]
SIMPLE_MOVE_LENGTH   = [1, 1, 0]
AUTHORIZED_LENGTHS = [ SIMPLE_MOVE_LENGTH, TAKE_MOVE_LENGTH,
                       CASTLING_MOVE_LENGTH ]

################################################################################

def diff_is_take(diff):
    # take move: one filled, none emptied, one changed
    return diff.length() == TAKE_MOVE_LENGTH

def diff_is_simple_move(diff):
    # simple move: one filled, one emptied, zero changed
    return diff.length() == SIMPLE_MOVE_LENGTH

def diff_sanity_check(diff):
    # sanity check on the board diff
    if diff.length() not in AUTHORIZED_LENGTHS:
        raise core.IllegalMove(
             'odd move(s) detected in diff: {}'.format(diff))

################################################################################

def read_castling(diff):
    if diff.length() != CASTLING_MOVE_LENGTH:
        return False

    for side, difflist in CASTLING_DIFF.items():
        if diff in difflist:
            return Castling(side)

    raise core.IllegalMove('odd move(s) detected in diff: {}'.format(diff))


def read(blind_board_diff):
    # aliases
    utils.log.debug("reading diff: {}".format(blind_board_diff))

    # sanity check: make sure this diff is not too odd...
    diff_sanity_check(blind_board_diff)

    # check for castling move
    castling_move = read_castling(blind_board_diff)
    if castling_move:
        utils.log.debug('read castling move {}'.format(castling_move))
        return castling_move

    # TODO check for promotion move

    # OK, so this oughta be a simple move or a take
    if diff_is_simple_move(blind_board_diff):
        utils.log.debug('diff is simple move')
    elif diff_is_take(blind_board_diff):
        utils.log.debug('diff is take move')
    else:
        utils.log.error('unreadable diff move')

    from_square = blind_board_diff.get_single_emptied()
    if diff_is_take(blind_board_diff):
        to_square = blind_board_diff.get_single_changed()
    else: # diff is simple move
        to_square = blind_board_diff.get_single_filled()

    move = Move(from_square, to_square)
    utils.log.debug('read move {}'.format(move))

    return move

