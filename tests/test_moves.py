import nose.tools
import chess

def test_from_string():
    test_cases = {
            'e2e4'    : chess.moves.Move('e2', 'e4', None, None),
            '1. e2e4' : chess.moves.Move('e2', 'e4', 1, chess.Color.WHITE)
        }

    for string, move in test_cases.items():
        nose.tools.eq_( chess.moves.from_string(string), move)

