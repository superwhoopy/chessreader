import cmd
import os
import sys

import capture, utils, tests, core
from imgprocessor import ImageProcessor


class Shell(cmd.Cmd):

    MOCK_OPTION = "mock"

    intro  = 'Welcome to chessreader shell! Type your command:\n'
    prompt = ' (chessreader) '

    def __init__(self):
        super(Shell, self).__init__()
        self.core = None
        self.capture_engine = None
        self.game_folder = None

    # TODO: make it a decorator
    def ensure_game_is_on(self):
        if core is None:
            utils.log.warn('no game currently active - run start first')
            return False
        return True

    def emptyline(self):
        pass

    def do_test(self, arg):
        'Run the chessreader test suite'
        utils.log.info('Running the test suite')
        tests.run()

    def do_start(self, arg):
        'Start a new game'
        if self.core:
            utils.log.error("A game is already on.")
            # TODO 'would you like to start a new one?'
            return
        utils.log.info("Starting new game!")

        self.game_folder = utils.create_new_game_folder()

        if arg == self.MOCK_OPTION:
            self.capture_engine = capture.Mock(tests.utils.collect_test_images())

        # TODO Should this be part of the core instead?
        empty = self.get_game_image(0)
        start = self.get_game_image(1)

        utils.log.info("Prepare the empty chessboard and press Enter...")
        input()
        self.capture_engine.capture(empty)
        utils.log.info("Prepare the chessboard in starting position and press Enter...")
        input()
        self.capture_engine.capture(start)
        utils.log.debug("Calibrating image processor...")
        self.core = core.Core(ImageProcessor(empty, start), self.capture_engine)
        utils.log.info("The game is on!")

    def do_show(self, arg):
        board_str = self.core.last_valid_board.__str__()
        print_str = "Next move: {0}\n".format(self.core.get_turn_str())
        utils.log.info(print_str + board_str)

    def do_takeback(self, arg):
        pass

    def do_capture(self, arg):
        '''Capture an image from the webcam'''
        capt_engine = capture.Fswebcam()
        capt_engine.capture()

    def do_next(self, arg):
        move, is_not_over = self.core.process_next_move()
        if not is_not_over:
            sys.exit(0)  # TODO a bit brutal

    def do_quit(self, arg):
        'Leave the shell and end the program'
        # TODO ask for confirmation if ongoing game
        utils.log.info("Bye!")
        sys.exit(0)

    def get_game_image(self, n):
        assert type(n) is int and n > -1
        return os.path.join(self.game_folder, "board-{0}.png".format(n))

