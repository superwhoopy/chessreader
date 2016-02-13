import cmd
import os
import sys

import capture, utils, tests, core
from core.main import IllegalMove
from imgprocessor import ImageProcessor


class Shell(cmd.Cmd):

    OPT_MOCK             = "mock"
    OPT_DELAYED_ANALYSIS = "delayed"

    intro  = 'Welcome to chessreader shell! Type your command:\n'
    prompt = ' (chessreader) '

    def __init__(self):
        super(Shell, self).__init__()
        self.core = None
        self.capture_engine = None
        self.game_folder = None

    def emptyline(self):
        pass

    def do_test(self, arg):
        'Run the chessreader test suite'
        utils.log.info('Running the test suite')
        tests.run()

    def do_start(self, arg):
        'Start a new game'
        utils.log.info("Starting new game!")

        # parse the options
        args_list = arg.split()
        if self.OPT_MOCK in args_list:
            # test_images = \
            #     tests.utils.collect_test_images('tests/pictures/game000')
            test_images = [ 'tests/pictures/game001/empty.jpg',
                            'tests/pictures/game001/start.jpg' ] + \
                    tests.utils.collect_test_images('tests/pictures/game001')
            capture_engine = capture.Mock(test_images)
        else:
            capture_engine = None
        do_live_analysis = self.OPT_DELAYED_ANALYSIS not in args_list

        # jump to the shell of the Core
        core_engine = core.Core(capture_engine, do_live_analysis)
        core_engine.cmdloop()

    def do_capture(self, arg):
        '''Capture an image from the webcam'''
        capt_engine = capture.Fswebcam()
        capt_engine.capture()

    def do_quit(self, arg):
        'Leave the shell and end the program'
        # TODO ask for confirmation if ongoing game
        utils.log.info("Bye!")
        sys.exit(0)

