import utils
from chess import Color
from chess.moves import Move

import pexpect
import time

class GnuChess:
    '''TODO'''

    DEFAULT_TIMEOUT = 1

    class Messages:
        ERR_ILLEGAL_MOVE = "Invalid move"

        RE_REGULAR_MOVE  = '(?P<move_count>\d+)\. '   \
                           '(?P<black_move>\.\.\. )?' \
                           '(?P<from_square>[abcdefgh][12345678])' \
                           '(?P<to_square>[abcdefgh][12345678])'

    @staticmethod
    def cmdline():
        binpath = pexpect.which('gnuchess')
        if binpath is None:
            utils.log.error('cannot find gnuchess on this system')

        args    = ['--xboard']

        return [binpath] + args

    def __init__(self, cmdline=cmdline.__func__(), do_log=True):
        # spawn the process
        self.proc = pexpect.spawn(' '.join(cmdline))
        self.proc.setecho(False)

        if do_log:
            self.logfile = open('gnuchess.log', 'wb')
            self.proc.logfile_read = self.logfile

        # make sure we have started
        self.proc.expect('Chess', timeout=self.DEFAULT_TIMEOUT)
        # set depth to minimum
        self.set_depth(1)

        # set to manual mode (i.e. 2 players)
        # self.proc.sendline('manual')

        # TODO: logging


    def read(self):
        # TODO: implement a "read all you can" method
        # for line in self.proc.read_stdout():
        #     utils.log.debug('gnuchess says "{}"'.format(line))
        pass

    def writeline(self, msg):
        utils.log.debug('writing "{}" to gnuchess stdin'.format(msg))
        self.proc.sendline(msg)

    def set_depth(self, depth):
        self.proc.sendline('depth {}'.format(depth))

    def play_move(self, move):
        # write the move to gnuchess stdin
        move_str = str(move)
        utils.log.debug("sending move '{}' to gnuchess...".format(move_str))
        self.proc.sendline(move_str)

        # then check stdout: make sure the move count is correct, plus we don't
        # want to find an invalid move...
        expect_good = '{}. {}'.format(move.move_count, move_str) \
                        if move.move_count is not None \
                        else move_str

        index = self.proc.expect([expect_good, self.Messages.ERR_ILLEGAL_MOVE,
                                  pexpect.TIMEOUT, pexpect.EOF],
                                 timeout=self.DEFAULT_TIMEOUT)

        if index == 0:
            # everything went well!
            utils.log.debug('gnuchess approved the move')
        elif index == 1:
            # illegal move
            utils.log.debug('illegal move reported by gnuchess')
        elif index in [2,3]:
            self.raise_error()



    def read_move(self):
        index = self.proc.expect([self.Messages.RE_REGULAR_MOVE,
                                    pexpect.TIMEOUT,
                                    pexpect.EOF],
                                 timeout=self.DEFAULT_TIMEOUT)
        if index == 0:
            # it's a match
            match = self.proc.match

            from_square = match.group('from_square').decode()
            to_square   = match.group('to_square').decode()
            color       = Color.BLACK if match.group('black_move') else \
                          Color.WHITE
            move_count  = int(match.group('move_count'))

            utils.log.debug(
                "read {} move from gnuchess '{}: {}{}'".format(
                    color, move_count, from_square, to_square))
            return Move(from_square, to_square, move_count, color)
        elif index in [1,2]:
            self.raise_error()



    def raise_error(self):
        utils.log.error(
            'an error occured with gnuchess: {}{}'.format(self.proc.before,
                                                          self.proc.after))

    def kill(self):
        self.writeline('quit')
