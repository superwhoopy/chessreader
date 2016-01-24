from .. import chessboard


def _build_conversion_dict():
    '''Builds `CONVERSION_DICT` global const var'''
    conversion_dict = {}
    for char in 'rnbqkp':
        conversion_dict[char] = chessboard.Color.BLACK
    for char in 'RNBQKP':
        conversion_dict[char] = chessboard.Color.WHITE
    conversion_dict['.'] = None
    return conversion_dict

'''Conversion dict: indexes are FEN characters, value is `None` for an empty
square, or one of `chessboard.Color` if the square is occuppied'''
CONVERSION_DICT = _build_conversion_dict()

################################################################################

def fen_2_blindboard(fen):
    '''Convert a full FEN-notation into a `BlindBoard` object

    The FEN string is expected to:

        - start with a carriage-return character
        - for each line, each square is represented with a character in
          `'rnbqkpRNBQKP.'`, followed by a space or a carriage-return (for the
          squares of the h-column.

    Example: the starting position is:

    ```

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

    ```

    Args:
        fen (str): full FEN-notation of a position, with the following

    Returns:
        A BlindBoard object matching the position
    '''
    # start by removing all these useless spaces
    fen = fen.replace(' ', '')
    # split into board lines
    fen = fen.split('\n')
    # remove first and last empty lines
    assert len(fen) == 10
    fen = fen[1:9]

    # dict of occupied square to be built
    occupied_squares = {}

    # browse the fen representation line by line
    for line in range(0, 8):
        fen_line = fen[line]
        assert len(fen_line) == 8
        # now for each character on this line
        for col in range(0,8):
            char        = fen_line[col]
            piece_color = CONVERSION_DICT[char]
            chessboard_col  = col
            chessboard_line = 7 - line
            square_name = \
                chessboard.board.square_name(chessboard_col, chessboard_line)
            if piece_color is not None:
                occupied_squares[square_name] = piece_color

    return chessboard.board.BlindBoard(occupied_squares)

def read_FEN_game(game):
    '''Convert a set of FEN position into a list of `BlindBoard` objects

    Args:
        game (str): concatenation of FEN positions as described in
            `fen_2_blindboard()`, with no separation.

    Returns:
        A list of `BlindBoard` objects representing the positions in order
    '''
    # 64 squares; for each squares 2 characters; one carriage return at the end
    # of each line; one additional carriage return as starting character
    len_one_board = 64*2 + 1

    blind_boards = []

    for i in range(0, len(game), len_one_board):
        position = game[i:i+len_one_board]
        blind_boards += [ fen_2_blindboard(position) ]

    return blind_boards
