import core
import chess.moves as moves
import chess.board as board
import utils

def diff_is_take(diff):
    pass

def diff_is_simple_move(diff):
    pass


def interpret_diff(blind_board_diff):
    # aliases
    emptied = blind_board_diff.emptied
    filled = blind_board_diff.filled
    changed = blind_board_diff.changed

    utils.log.debug("diffing boards: {}".format(blind_board_diff))

    # sanity check on the board diff
    AUTHORIZED_LENGTHS = [
            [1, 1, 0], # simple move: 1 square cleared, 1 filled
            [1, 0, 1], # take: 1 square cleared, 1 changed
            [2, 2, 0], # castling
        ]
    if [len(emptied), len(filled), len(changed)] not in AUTHORIZED_LENGTHS:
        raise core.IllegalMove(
             'too many pieces seem to have moved: {}'.format(blind_board_diff))

    # check for castling
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

    # promoting move
    # TODO

    return move

