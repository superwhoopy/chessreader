'''Pieces moves representation'''

from enum import Enum
import re

import chess
from chess import Color
import utils

################################################################################

RE_MOVE = re.compile( '((?P<move_count>\d+)\. )?'   \
                      '(?P<black_move>\.\.\. )?'    \
                      '(?P<from_square>[abcdefgh][12345678])' \
                      '(?P<to_square>[abcdefgh][12345678])')


def from_string(string):
    match = RE_MOVE.match(string)
    if not match:
        # TODO: throw exception instead
        utils.log.error(
                "string '{}' does not match a valid move".format(string))

    from_square = match.group('from_square')
    to_square   = match.group('to_square')
    move_count  = match.group('move_count')

    color       = None
    if move_count:
        color = Color.BLACK if match.group('black_move') else Color.WHITE

    return Move(match.group('from_square'), match.group('to_square'),
                move_count, color)

################################################################################


class Move:
    '''Default simple move, from a square to another'''

    def __init__(self, from_square, to_square, move_count=None, color=None):
        assert from_square in chess.board.ALL_SQUARES
        assert to_square   in chess.board.ALL_SQUARES

        self.from_square = from_square
        self.to_square   = to_square
        self.move_count  = move_count
        self.color       = color


    def __str__(self):
        return '{}{}'.format(self.from_square, self.to_square)


    def pretty_print(self):
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
        assert side in Castling.Side
        self.from_square = 'O-O'
        self.to_square   = '-O' if side == Castling.Side.QUEEN \
                                else ''

        self.move_count  = move_count
        self.color       = color

# TODO: PromotionMove? EnPassant?
