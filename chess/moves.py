'''Pieces moves representation'''

from enum import Enum
import re

import chess
from chess import Color
import utils

################################################################################

RE_MOVE = re.compile( r'((?P<move_count>\d+)\. )?'   \
                      r'(?P<black_move>\.\.\. )?'    \
                      r'(?P<from_square>[abcdefgh][12345678])' \
                      r'(?P<to_square>[abcdefgh][12345678])')
'''Regexp parsing a move.

Two kinds of strings are recognized:

    - "naked" string made only of two consecutive square names;
    - "complete" string made of a move count, and a color indication. E.g:
      `1. ... e7e5` to tell that Black plays e7e5 on their first move.
'''


def from_string(string):
    match = RE_MOVE.match(string)

    # TODO: recognize castling?

    if not match:
        # TODO: throw exception instead?
        utils.log.error(
                "string '{}' does not match a valid move".format(string))

    from_square = match.group('from_square')
    to_square   = match.group('to_square')
    move_count  = match.group('move_count')

    color       = None
    if move_count:
        color = Color.BLACK if match.group('black_move') else Color.WHITE

    return Move(from_square, to_square, move_count, color)

################################################################################


class Move:
    '''Default simple move, from a square to another'''

    def __init__(self, from_square, to_square, move_count=None, color=None):
        '''Create a default move, other than castling

        Args:
            from_square (str): square the move starts from
            to_square   (str): square the move goes to
            move_count  (int): optional argument, how manu moves occurred before
               in the game; recall that this value starts at 1 for the first
               move, and is incremented every two moves (one for White, one for
               Black)
            color    `Color`): optional, `Color.WHITE` or `Color.BLACK` to tell
               who plays this move
        '''
        assert from_square in chess.board.ALL_SQUARES
        assert to_square   in chess.board.ALL_SQUARES

        self.from_square = from_square
        self.to_square   = to_square
        self.move_count  = int(move_count) if move_count is not None else None
        self.color       = color


    def __str__(self):
        return '{}{}'.format(self.from_square, self.to_square)


    def pretty_print(self):
        '''Pretty-print the move, including move count & color

        If the move comes with a `move_count` and a `color`, print these out.
        For instance, if Black plays e7e5 as first move, this method will
        return `2. ... e7e5` whereas `__str__` only returns `e7e5`.
        '''
        move_count_str = ''
        color_str      = ''

        if self.move_count is not None:
            move_count_str = '{}. '.format(self.move_count)
        if self.color == chess.Color.BLACK:
            color_str = '... '

        return move_count_str + color_str + self.__str__()


    def __eq__(self,other):
        return self.from_square == other.from_square and \
               self.to_square   == other.to_square   and \
               self.color       == other.color       and \
               self.move_count  == other.move_count



class Castling(Move):
    '''Castling move'''

    class Side(Enum):
        '''Side of the castling move'''
        KING  = 0
        QUEEN = 1

    def __init__(self, side, move_count=None, color=None):
        '''Create a Castling move

        Args:
            side       : side of the castling move : `Side.KING` or `Side.QUEEN`
            move_count : see `Move.__init__()`
            color      : see `Move.__init__()`
        '''
        assert side in Castling.Side
        self.from_square = 'O-O'
        self.to_square   = '-O' if side == Castling.Side.QUEEN \
                                else ''

        self.move_count  = move_count
        self.color       = color


class Promote(Move):
    '''Promotion move'''

    def __init__(self, from_square, to_square, promote_to,
                 move_count=None, color=None):
        '''TODO'''
        # sanity checks
        assert promote_to in chess.Piece
        expected_line = {
                chess.Color.WHITE : [7],
                chess.Color.BLACK : [0],
                None              : [0, 7],
                }
        _, line = chess.board.square_coordinates(to_square)
        assert line in expected_line

        # call parent ctor
        super(Promote, self).__init__(from_square, to_square, move_count,
                color)

        # register the piece to promote to
        self.promote_to = promote_to

    def __eq__(self, other):
        return super(Promote, self).__eq__(self,other) and \
               self.promote_to == other.promote_to

    def __str__(self):
        return super(Promote, self).__str__(self) + \
               '={}'.format(self.promote_to.value)
