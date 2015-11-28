import nose.tools

import core
import engine

import tests.utils
from tests.fen_games.opera import GAME

def test_opera():
    # start by building all the blind boards of the Game
    blind_boards = tests.utils.read_FEN_game(GAME)

    # build the list of moves
    moves = [ core.diffreader.read(to_position.diff(from_position))
                for from_position, to_position in zip(blind_boards[0:-1],
                                                      blind_boards[1:]) ]

    # call gnuchess and play the game! everything should go fine...
    my_engine = engine.GnuChess()
    for move in moves:
        my_engine.play_move(move)

    # save the game and quit
    my_engine.writeline("pgnsave opera_game.pgn")
    my_engine.kill()
