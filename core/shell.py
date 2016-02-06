import cmd
import os
import sys

import capture, utils, tests, core
from core.main import IllegalMove
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
        utils.log.info("Starting new game!")

        if arg == self.MOCK_OPTION:
            capture_engine = capture.Mock(tests.utils.collect_test_images())
        else:
            capture_engine = None

        # jump to the shell of the Core
        core.Core(capture_engine).cmdloop()


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
        try:
            move, is_not_over = self.core.process_next_move()
            if not is_not_over:
                sys.exit(0)  # TODO a bit brutal
        except IllegalMove as e:
            utils.log.warn("Recieved illegal move: {0}".format(e))

    def do_quit(self, arg):
        'Leave the shell and end the program'
        # TODO ask for confirmation if ongoing game
        utils.log.info("Bye!")
        sys.exit(0)

    def get_game_image(self, n):
        assert type(n) is int and n > -1
        return os.path.join(self.game_folder, "board-{:03}.png".format(n))

