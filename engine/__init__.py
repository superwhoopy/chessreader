import utils

import pexpect

class GnuChess:
    '''TODO'''

    GNUCHESS_CMDLINE = 'gnuchess --xboard'

    def __init__(self):
        self.proc = pexpect.spawn(self.GNUCHESS_CMDLINE)

        self.readline()
        self.write('depth 1')
        self.proc.expect('depth of ')
        self.readline()
        self.readline()
        self.readline()

    def readline(self):
        line = self.proc.readline()
        utils.log.debug('gnuchess says: "{}"'.format(line))

    def write(self, msg):
        utils.log.debug('write to gnuchess: "{}"'.format(msg))
        self.proc.sendline(msg)

    def play_move(self, player, move):
        pass

    def read_move(self):
        pass

    def kill(self):
        self.proc.sendline('quit')
