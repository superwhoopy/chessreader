import utils

import subprocess

class Generic:
    '''TODO'''

    GNUCHESS_PATH = 'gnuchess'

    def __init__(self):
        try:
            self.process = subprocess.Popen(
                    [self.GNUCHESS_PATH],
                    universal_newlines = True,
                    stdin = subprocess.PIPE,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE)
        except (subprocess.SubprocessError, OSError) :
            utils.log.error('unable to fork {}'.format(self.GNUCHESS_PATH))

        self.read()
        self.write('depth 1\n')
        self.read()
        self.write('e2e4\n')
        self.read()

    def read(self):
        utils.log.debug(self.process.stdout.read())

    def write(self, msg):
        utils.log.debug('writing "{}" to stdout'.format(msg))
        self.process.stdin.write(msg)

    def play_move(self, player, move):
        pass

    def read_move(self):
        pass

    def kill(self):
        self.process.communicate('quit')
