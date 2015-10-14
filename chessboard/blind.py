import chessboard

def find_moves(from_board, to_board):
    '''TODO
    '''
    emptied_squares = from_board.occupied_squares - to_board.occupied_squares
    filled_squares  = to_board.occupied_squares - from_board.occupied_squares
    return emptied_squares, filled_squares


class BlindChessboard:
    '''TODO
    '''

    occupied_squares = set()

    def __init__(self, occupied_squares=None):
        if occupied_squares is None:
            return
        for square in occupied_squares:
            assert square in chessboard.ALL_SQUARES
        self.occupied_squares = occupied_squares

    def clear(self):
        self.occupied_squares = set()

    def add_piece(self, square_name):
        assert square_name in chessboard.ALL_SQUARES
        self.occupied_squares.add(square_name)
