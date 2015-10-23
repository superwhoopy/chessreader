import cmd
import sys
from enum import Enum

import utils.log
import tests
import core

class CaptureShell(cmd.Cmd):
    prompt = ' (capture) '

    core   = core.Core()

    def do_read(self, arg):
        'Capture a picture of the board and process it'
        self.core.run()

    def do_end(self, arg):
        'End capturing mode and return to main shell'
        utils.log.info("leaving capture mode")
        return True




class Shell(cmd.Cmd):

    class State(Enum):
        BASE    = 0
        CAPTURE = 1

    intro  = 'Welcome to chessreader shell! Type your command:\n'
    prompt = ' (chessreader) '

    state = State.BASE

    def emptyline(self):
        pass

    def do_test(self, arg):
        'Run the chessreader test suite'
        utils.log.info('running the test suite')
        tests.run()

    def do_start(self, arg):
        'Start a new game'
        CaptureShell().cmdloop()

    def do_quit(self, arg):
        'Leave the shell and end the program'

        utils.log.info("quitting; bye!")
        sys.exit(0)

