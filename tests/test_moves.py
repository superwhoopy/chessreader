import nose.tools
import chess

def test_from_string():
    test_cases = {
            'e2e4'         : chess.moves.Move('e2', 'e4', None, None),
            '1. e2e4'      : chess.moves.Move('e2', 'e4', 1, chess.Color.WHITE),
            '10. ... h8a1' : chess.moves.Move('h8', 'a1', 10, chess.Color.BLACK)
        }

    for string, expected_move in test_cases.items():
        move = chess.moves.from_string(string)
        nose.tools.eq_(move, expected_move)

