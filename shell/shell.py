import cmd
import sys

import capture, utils, tests, core


class Shell(cmd.Cmd):

    intro  = 'Welcome to chessreader shell! Type your command:\n'
    prompt = ' (chessreader) '

    core = None

    # TODO: make it a decorator
    def ensure_game_is_on(self):
        if core is None:
            utils.log.warn('no game currently active - run start first')
            return False
        return True

    def emptyline(self):
        self.do_read("")

    def do_test(self, arg):
        'Run the chessreader test suite'
        utils.log.info('running the test suite')
        tests.run()

    def do_start(self, arg):
        'Start a new game'
        self.core = core.Core()

    def do_takeback(self, arg):
        pass

    def do_capture(self, arg):
        '''Capture an image from the webcam'''
        capt_engine = capture.Fswebcam()
        capt_engine.capture()

    def do_read(self, arg):
        self.core.run()

    def do_quit(self, arg):
        'Leave the shell and end the program'
        utils.log.info("quitting; bye!")
        sys.exit(0)

