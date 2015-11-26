import chess


def _build_conversion_dict():
    conversion_dict = {}
    for char in 'rnbqkp':
        conversion_dict[char] = chess.Color.BLACK
    for char in 'RNBQKP':
        conversion_dict[char] = chess.Color.WHITE
    conversion_dict['.'] = None
    return conversion_dict

CONVERSION_DICT = _build_conversion_dict()

################################################################################

def fen_2_blindboard(fen):
    # start by removing all these useless spaces
    fen = fen.replace(' ', '')
    # split into board lines
    fen = fen.split('\n')
    # remove first and last empty lines
    assert len(fen)==10
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
                chess.board.square_name(chessboard_col, chessboard_line)
            if piece_color is not None:
                occupied_squares[square_name] = piece_color

    return chess.board.BlindBoard(occupied_squares)

def read_FEN_game(game):
    # 64 squares; for each squares 2 characters; one carriage return at the end
    # of each line; one additional carriage return as starting character
    len_one_board = 64*2 + 1

    blind_boards = []

    for i in range(0, len(game), len_one_board):
        position = game[i:i+len_one_board]
        blind_boards += [ fen_2_blindboard(position) ]

    return blind_boards
