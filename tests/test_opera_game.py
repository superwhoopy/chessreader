import os

import nose.tools

import core.diffreader
from .utils import read_boards_from_pgn, read_moves_from_pgn


def test_opera():
    '''Test deduction of all moves from the Opera Game'''
    # start by building all the blind boards of the Game

    opera_game = os.path.join(os.path.split(__file__)[0], 'games/opera.pgn')
    # read the boards from the PGN and turn them into BlindBoards
    blind_boards = list(read_boards_from_pgn(opera_game, use_blindboards=True))

    # deduce the list of moves from the BlindBoards
    deduced_moves = [ core.diffreader.read(to_position.diff(from_position))
                      for from_position, to_position in zip(blind_boards[0:-1],
                                                      blind_boards[1:])
    ]

    # read the list of actual moves from the PGN and compare it to the deduced moves
    moves = list(read_moves_from_pgn(opera_game))
    nose.tools.eq_(moves, deduced_moves)

    # TODO play the game with gnuchess?
