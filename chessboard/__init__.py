import string

COL_NAMES = string.ascii_lowercase[:8]
ROW_NAMES = range(1, 9)

ALL_SQUARES = [ '{}{}'.format(col, row) for col in COL_NAMES \
                                        for row in ROW_NAMES ]

class SquareOutOfBounds(Exception):
    pass

class MalformedSquareName(Exception):
    pass

################################################################################

def square_name(x_pos, y_pos):
    name = "{}{}".format(COL_NAMES[x_pos], y_pos+1)
    if name not in ALL_SQUARES:
        raise SquareOutOfBounds
    return name

def square_coordinates(square):
    if not len(square) == 2:
        raise MalformedSquareName

    col_name = square[0]
    if col_name not in COL_NAMES:
        raise SquareOutOfBounds
    x_pos = COL_NAMES.index(col_name)

    y_pos = square[1] - 1
    if y_pos >= len(ROW_NAMES) or y_pos < 0:
        raise SquareOutOfBounds

    return x_pos, y_pos

