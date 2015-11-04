import utils

import subprocess

class GnuChess:
    '''TODO'''

    GNUCHESS_CMDLINE = ['gnuchess', '--xboard']

    def __init__(self):
        try:
            self.process = subprocess.Popen(
                    self.GNUCHESS_CMDLINE,
                    universal_newlines = True,
                    stdin = subprocess.PIPE,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE)
        except (subprocess.SubprocessError, OSError) :
            utils.log.error('unable to fork gnuchess')

        self.readline()
        self.write('depth 1\n')
        self.readline()
        self.write('e2e4\n')
        self.readline()

    def readline(self):
        line = self.process.stdout.readline()
        utils.log.debug('gnuchess says "{}"'.format(line))

    def write(self, msg):
        utils.log.debug('writing "{}" to stdout'.format(msg))
        self.process.stdin.write(msg)

    def play_move(self, player, move):
        pass

    def read_move(self):
        pass

    def kill(self):
        self.process.stdin.write('quit')
