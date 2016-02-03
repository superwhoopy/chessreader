'''
The diffreader module main purpose is to turn BlindBoard.Diff objects into valid
chess.Move objects, or to throw an error if the move is invalid.
'''
import chess
from chess import Move

from chessboard import BlindBoard
import utils

class IllegalMove(Exception):
    pass

# list of blindboard diffs for castling moves, and the corresponding moves (as
# encoded in UCI)
CASTLING_DIFFS = [
    # king-side
    (BlindBoard.Diff(chess.BB_E1 | chess.BB_H1, chess.BB_F1 | chess.BB_G1, 0),
        Move(chess.E1, chess.G1)),
    (BlindBoard.Diff(chess.BB_E8 | chess.BB_H8, chess.BB_F8 | chess.BB_G8, 0),
        Move(chess.E8, chess.G8)),

    # queen-side
    (BlindBoard.Diff(chess.BB_E1 | chess.BB_A1, chess.BB_C1 | chess.BB_D1, 0),
        Move(chess.E1, chess.C1)),
    (BlindBoard.Diff(chess.BB_E8 | chess.BB_A8, chess.BB_C8 | chess.BB_D8, 0),
        Move(chess.E8, chess.C8))
]


class DiffLength:
    CASTLING = (2, 2, 0)
    TAKE     = (1, 0, 1)
    SIMPLE   = (1, 1, 0)

    # TODO: find a pythonic way to do this...
    valid    = { CASTLING, TAKE, SIMPLE }

################################################################################


def _diff_is_take(diff):
    # take move: one filled, none emptied, one changed
    return diff.length() == DiffLength.TAKE


def _diff_is_simple_move(diff):
    # simple move: one filled, one emptied, zero changed
    return diff.length() == DiffLength.SIMPLE


def _diff_sanity_check(diff):
    # sanity check on the board diff
    if diff.length() not in DiffLength.valid:
        raise IllegalMove(
             'odd move(s) detected in diff: {} (unexpected diff length: {})'
             .format(diff, diff.length())
        )

################################################################################


def _read_castling(diff):
    if diff.length() != DiffLength.CASTLING:
        return False

    for castling_diff, castling_move in CASTLING_DIFFS:
        if castling_diff == diff:
            return castling_move

    raise IllegalMove('odd move(s) detected in diff: {} '
           '(the move has the length of a castling but does not match any)'
           .format(diff)
    )


def read(blind_board_diff):
    # aliases
    utils.log.debug("reading diff: {}".format(blind_board_diff))

    # sanity check: make sure this diff is not too odd...
    _diff_sanity_check(blind_board_diff)

    # check for castling move
    castling_move = _read_castling(blind_board_diff)
    if castling_move:
        utils.log.debug('read castling move {}'.format(castling_move))
        return castling_move

    # TODO check for promotion move
    # TODO check for 'en passant'

    # OK, so this oughta be a simple move or a take: print out some debug
    # message
    if _diff_is_simple_move(blind_board_diff):
        utils.log.debug('diff is simple move')
    elif _diff_is_take(blind_board_diff):
        utils.log.debug('diff is take move')
    else:
        utils.log.error('unreadable diff move')

    # get the square the move came from, the square it goes to, create the
    # appropriate Move object and return it

    from_square = utils.singleton_get(blind_board_diff.get_emptied())
    if _diff_is_take(blind_board_diff):
        to_square = utils.singleton_get(blind_board_diff.get_changed())
    else: # diff is simple move
        to_square = utils.singleton_get(blind_board_diff.get_filled())

    move = Move(from_square, to_square)
    utils.log.debug('read move {}'.format(move))

    return move

