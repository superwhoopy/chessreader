import os
import re
import nose

import chess.pgn
from chess import Board, Piece, BLACK, WHITE, PAWN

from chessboard import BlindBoard


def boards_from_pgn(pgn_path, use_blindboards=False):
    '''TODO'''
    with open(pgn_path, 'r') as pgn_file:
        def _recurse(node):
            yield BlindBoard.from_board(node.board()) if use_blindboards else \
                  node.board()

            if node.is_end():
                return
            assert len(node.variations) <= 1
            yield from _recurse(node.variations[0])

        root_node = chess.pgn.read_game(pgn_file)
        yield from _recurse(root_node)


def read_moves_from_pgn(pgn_path):
    '''TODO'''
    with open(pgn_path, 'r') as pgn_file:
        node = chess.pgn.read_game(pgn_file)
        while node.variations:
            next_node = node.variations[0]
            yield next_node.move
            node = next_node


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


@nose.tools.nottest
def collect_test_images(dir_path):
    image_regex = re.compile('board-[0-9]{3}-[0-1]\.jpg')
    images = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
              if image_regex.match(f)]
    return natural_sort(images)  # board-001-0.jpg, board-001-1.jpg, ...
