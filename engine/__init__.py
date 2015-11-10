import utils
from chess import Color
from chess.moves import Move

import pexpect
import time

class GnuChess:
    '''TODO'''

    GNUCHESS_CMDLINE = ['gnuchess', '--xboard']
    TIMEOUT = 1

    ERRMSG_ILLEGAL_MOVE = "Invalid move"

    proc = pexpect.spawn(' '.join(GNUCHESS_CMDLINE))



    def __init__(self):
        # make sure we have started
        self.proc.expect('Chess', timeout=self.TIMEOUT)
        # set to manual mode (i.e. 2 players)
        self.proc.sendline('manual')


    def read(self):
        for line in self.proc.read_stdout():
            utils.log.debug('gnuchess says "{}"'.format(line))

    def writeline(self, msg):
        utils.log.debug('writing "{}" to stdin'.format(msg))
        self.proc.writeline(msg)

    def play_move(self, move):
        # write the move to gnuchess stdin
        self.proc.sendline(move)

        # then check stdout: make sure the move count is correct, plus we don't
        # want to find an invalid move...
        expect_good = '{}. {}'.format(move.move_count, str(move)) \
                        if move.move_count is not None \
                        else str(move)
        index = self.proc.expect([expect_good, self.ERRMSG_ILLEGAL_MOVE,
                                  pexpect.TIMEOUT, pexpect.EOF])

        if index == 0:
            # everything went well!
            pass
        elif index == 1:
            # illegal move
            pass
        elif index in [2,3]:
            utils.log.error(
                'an error occured with gnuchess: {}'.format(self.proc.after))

    def read_move(self):
        MOVE_RE = '(?P<move_count>\d+)\. ' \
                  '(?P<from_square>[abcdefgh][12345678])' \
                  '(?P<to_square>[abcdefgh][12345678])'

        index = self.proc.expect(MOVE_RE, pexpect.TIMEOUT, pexpect.EOF)
        if index == 0:
            # it's a match
            match = self.proc.match
            return Move(match.group('from_square'),
                        match.group('to_square'),
                        int(match.group('move_count')))
        elif index in [1,2]:
            pass

    def kill(self):
        self.writeline('quit')
