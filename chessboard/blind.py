import chessboard

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

    def __eq__(self, other):
        return self.occupied_squares == other.occupied_squares

    def __sub__(self, other):
        emptied_squares = other.occupied_squares - self.occupied_squares
        filled_squares  = self.occupied_squares - other.occupied_squares
        return emptied_squares, filled_squares

    def clear(self):
        self.occupied_squares = set()

    def add_piece(self, square_name):
        assert square_name in chessboard.ALL_SQUARES
        self.occupied_squares.add(square_name)
