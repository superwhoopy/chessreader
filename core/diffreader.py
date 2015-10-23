import core
import chess.moves as moves
import chess.board as board
import utils

def diff_is_take(diff):
    # take move: one filled, none emptied, one changed
    return diff.length() == [1, 0, 1]

def diff_is_simple_move(diff):
    # simple move: one filled, one emptied, zero changed
    return diff.length() == [1, 1, 0]

def diff_sanity_check(diff):
    # sanity check on the board diff
    AUTHORIZED_LENGTHS = [
            [1, 1, 0], # simple move: 1 square cleared, 1 filled
            [1, 0, 1], # take: 1 square cleared, 1 changed
            [2, 2, 0], # castling
        ]
    if diff.length() not in AUTHORIZED_LENGTHS:
        raise core.IllegalMove(
             'too many pieces seem to have moved: {}'.format(diff))

def diff_is_castling(diff):
    pass

################################################################################

def read(blind_board_diff):
    # aliases
    emptied = blind_board_diff.emptied
    filled = blind_board_diff.filled
    changed = blind_board_diff.changed

    utils.log.debug("diffing: {}".format(blind_board_diff))

    # sanity check: make sure this diff is not too odd...
    diff_sanity_check(blind_board_diff)

    # check for castling move
    CASTLING_MOVES = {
        board.BlindBoard.Diff({'e1', 'h1'}, {'f1','g1'}, {}) :
            moves.Castling(moves.Castling.Side.King),
        board.BlindBoard.Diff({'e8', 'h8'}, {'f8','g8'}, {}) :
            moves.Castling(moves.Castling.Side.King),
        board.BlindBoard.Diff({'e1', 'a1'}, {'c1','d1'}, {}) :
            moves.Castling(moves.Castling.Side.Queen),
        board.BlindBoard.Diff({'e8', 'a8'}, {'c8','d8'}, {}) :
            moves.Castling(moves.Castling.Side.Queen),
    }
    if blind_board_diff in CASTLING_MOVES:
        utils.log.debug("castling move detected")
        return CASTLING_MOVES[blind_board_diff]

    # check for promotion move
    # TODO

    # OK, this oughta be a simple move or a take
    assert diff_is_simple_move(blind_board_diff) or \
           diff_is_take(blind_board_diff)

    from_square = emptied.pop()
    to_square   = changed.pop() if diff_is_take(blind_board_diff) else \
                  filled.pop()

    return moves.Move(from_square, to_square)

