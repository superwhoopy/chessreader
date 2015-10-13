class BlindChessboard:
    '''TODO
    '''

    occupied_squares = set()

    def __init__(self, occupied_squares=None):
        if occupied_squares is not None:
            self.occupied_squares = occupied_squares

    def find_moves(self, to_board):
        '''TODO
        '''
        emptied_squares = self.occupied_squares - to_board.occupied_squares
        filled_squares  = to_board.occupied_squares - self.occupied_squares
        return emptied_squares, filled_squares

