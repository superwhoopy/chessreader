import utils

import subprocess

class GnuChess:
    '''TODO'''

    GNUCHESS_CMDLINE = ['gnuchess', '--xboard']

    proc = utils.Proc(GNUCHESS_CMDLINE)

    def __init__(self):
        self.read()
        self.writeline('depth 1')

    def read(self):
        for line in self.proc.read_stdout():
            utils.log.debug('gnuchess says "{}"'.format(line))

    def writeline(self, msg):
        utils.log.debug('writing "{}" to stdin'.format(msg))
        self.proc.writeline(msg)

    def play_move(self, player, move):
        pass

    def read_move(self):
        pass

    def kill(self):
        self.writeline('quit')
