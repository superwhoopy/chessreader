import cmd
import sys
from enum import Enum

import utils.log
import tests
import core
import engine


class Shell(cmd.Cmd):

    intro  = 'Welcome to chessreader shell! Type your command:\n'
    prompt = ' (chessreader) '

    def emptyline(self):
        pass

    def do_test(self, arg):
        'Run the chessreader test suite'
        utils.log.info('running the test suite')
        tests.run()

    def do_start(self, arg):
        'Start a new game using GNU chess'
        self.gnuchess = engine.GnuChess()

    def do_read(self, arg):
        self.gnuchess.read()

    def do_write(self, arg):
        self.gnuchess.writeline(arg)

    def do_quit(self, arg):
        'Leave the shell and end the program'

        utils.log.info("quitting; bye!")
        sys.exit(0)

