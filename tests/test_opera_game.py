import nose.tools

import core
import tests.utils
from tests.fen_games.opera import GAME

def test_opera():
    # start by building all the blind boards of the Game
    blind_boards = tests.utils.read_FEN_game(GAME)
    first_move_diff = blind_boards[1].diff(blind_boards[0])
    move = core.diffreader.read(first_move_diff)
    print(move)

    # diff the first move

