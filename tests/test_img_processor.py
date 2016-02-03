import os
import re

import chess
from chess import Piece, PAWN, WHITE, BLACK

from chessboard.board import BlindBoard
from .utils import natural_sort
from imgprocessor import ImageProcessor


def compare_blindboards(expected, actual, file_name=None):
    if not expected == actual:
        info = ""
        if file_name:
            info = "for `%s`" % os.path.basename(file_name)
        print("BlindBoards are different %s: expected" % info)
        print(expected)
        print("But found instead:")
        print(actual)
        raise AssertionError("BlindBoards are different")
    return

# TODO store the game as a PGN instead ?
def expected_boards():
    '''This generator returns all the BlindBoards corresponding to the
    images inside ./pictures (starting with board-2.jpg)'''
    b = BlindBoard.get_starting_board()
    b.move_piece(chess.E2, chess.E4)
    yield b  # board-2.jpg
    b.move_piece(chess.E7, chess.E5)
    yield b
    b.move_piece(chess.G1, chess.F3)
    yield b
    b.move_piece(chess.B8, chess.C6)
    yield b
    b.move_piece(chess.F1, chess.B5)
    yield b
    b.move_piece(chess.G8, chess.F6)
    yield b
    b.move_piece(chess.E1, chess.F1)
    b.move_piece(chess.H1, chess.G1)
    yield b
    b.remove_piece_at(chess.F6)
    b.change_color_at(chess.E4)
    yield b
    b.move_piece(chess.D1, chess.E2)
    yield b
    b.move_piece(chess.D7, chess.D5)
    yield b
    b.remove_piece_at(chess.F3)
    b.change_color_at(chess.E5)
    yield b
    b.move_piece(chess.C8, chess.D7)
    yield b
    b.remove_piece_at(chess.E5)
    b.change_color_at(chess.C6)
    yield b
    b.remove_piece_at(chess.B7)
    b.change_color_at(chess.C6)
    yield b
    b.move_piece(chess.B5, chess.D3)
    yield b

def test_imgage_processor():
    '''Test image processor'''

    expected_board = BlindBoard.get_starting_board()
    expected_board.remove_piece_at(chess.E2)
    expected_board.set_piece_at(chess.E4, Piece(PAWN, WHITE))

    # retrieve all the images paths and sort
    image_regex = re.compile('board-[0-9]+\.jpg')
    pictures_folder = os.path.join(os.path.split(__file__)[0], 'pictures')
    images = [os.path.join(pictures_folder, f) for f in os.listdir(pictures_folder)
              if image_regex.match(f)]
    images = natural_sort(images)  # board-0.jpg, board-1.jpg, ...

    print("\n  * Calibrating image processor...")
    processor = ImageProcessor(images[0], images[1])

    for img, expected_board in zip(images[2:], expected_boards()):
        print("  * Processing `%s`..." % os.path.basename(img))
        processor.process(img)
        board = processor.get_blindboard()
        compare_blindboards(expected_board, board, img)
