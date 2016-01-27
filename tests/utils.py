
import chess.pgn
from chess import Board, Piece, BLACK, WHITE, PAWN

from ..chessboard.board import BlindBoard


def read_boards_from_pgn(pgn_path, use_blindboards=False):
    '''
    Generator function that takes as input the path to a PGN file
    and returns an iterator over `Board` or `BlindBoard` objects describing the game
    '''
    def _convert(board):
        if use_blindboards:
            return board_to_blindboard(board)
        else:
            return board

    with open(pgn_path, 'r') as pgn_file:
        node = chess.pgn.read_game(pgn_file)
        while node.variations:
            yield _convert(node.board())
            node = node.variation(0)
        yield _convert(node.board())


def read_moves_from_pgn(pgn_path):
    with open(pgn_path, 'r') as pgn_file:
        node = chess.pgn.read_game(pgn_file)
        while node.variations:
            next_node = node.variation(0)
            yield next_node.move
            node = next_node


def board_to_blindboard(board):
    '''
    Takes as input a `Board` object and makes it 'blind' by turning all pieces into pawns.
    '''
    assert isinstance(board, Board)
    blindboard = BlindBoard()

    for color in (BLACK, WHITE):
        # occupied_pieces is a set of integers
        occupied_squares = BlindBoard.Diff.get_squares_from_mask(board.occupied_co[color])
        for square in occupied_squares:
            blindboard.set_piece_at(square, Piece(PAWN, color))

    return blindboard
