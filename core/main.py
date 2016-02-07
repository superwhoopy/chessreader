import chess, cmd, os
from chessboard import BlindBoard

from imgprocessor import ImageProcessor
import capture, utils
from utils import log
import core.diffreader
from core.diffreader import IllegalMove

#
# TODO-list for this module:
#
#   - implement post-mortem analysis of the game in do_analyze() when
#     do_live_analysis is False
#
#   - implement do_takeback()
#
#   - misc. options & parameters implementation - through a dedicated command?
#     do_set()?
#        - add mandatory parameter for camera orientation
#        - implement 'do_keep_trace_file' option: possibly clear the directory
#          upon leaving - implement __enter__() and __exit__() methods?
#        - enable/disable do_show_each_move dynamically
#

class Core(cmd.Cmd):
    '''TO-FUCKING-DO'''

    prompt = ' (chessreader-PLAYING) '

    def __init__(self, capture_engine=None, do_live_analysis=True):
        # parent ctor
        super(Core, self).__init__()
        # core tools
        self.capture_engine        = capture_engine or capture.Fswebcam()
        self.image_processor       = None # will be init. in preloop()
        self.chess_engine          = None  # TODO
        # game status
        self.last_valid_blindboard = BlindBoard.get_starting_board()
        self.current_board         = chess.Board()
        # core options
        self.game_folder           = utils.create_new_game_folder()
        self.do_keep_trace_files   = True # TODO
        self.do_live_analysis      = do_live_analysis
        self.do_show_each_move     = True # TODO

    # cmd-related stuff ########################################################
    #
    def emptyline(self):
        self.do_next(None)

    def preloop(self):
        '''TODO'''
        # whether live analysis is on or off, perform calibration now
        self._calibration()
        log.info("The game is on!")

    def do_next(self, _):
        '''TODO'''
        # make sure the game is still on
        if self.current_board.is_game_over():
            log.warn('Game is over! Type "help" to see authorized commands')

        # img capture!
        img_path = self._get_img_name()
        self.capture_engine.capture(img_path)
        assert os.path.exists(img_path)

        if self.do_live_analysis:
            try:
                self._process_next_move(img_path)
            except IllegalMove as e:
                log.warn(str(e))
                log.info('no move registered; type "next" again to retry')
        else:
            self.current_board.push(chess.Move.null())

    def do_show(self, _):
        '''Show the current status of the board in a human-readable fashion'''
        print_str = "\nNext move: {0}\n".format(self._get_turn_str())
        utils.log.info(print_str + str(self.current_board) + '\n')

    def do_analyze(self, _):
        '''End the current game and run analysis on captured images'''
        if self.do_live_analysis:
            log.warn('live analysis is on: nothing to do')
            return
        # consider the game is over, and analyze all the pictures available
        # TODO

    def do_takeback(self, _):
        '''TODO'''
        pass

    def do_quit(self, _):
        '''End the current game and leave'''
        if utils.confirm_yes_no('Are you sure you want to end this game?'):
            log.info('leaving current game')
            return True

    ############################################################################

    def _calibration(self):
        # TODO: this is messy - we need to handle calibration failure!
        empty_img = os.path.join(self.game_folder, 'empty.jpg')
        start_img = self._get_img_name()

        log.info("Prepare the empty chessboard and press Enter")
        input()
        self.capture_engine.capture(empty_img)
        log.info("Prepare the chessboard in starting position "
                 "and press Enter")
        input()
        self.capture_engine.capture(start_img)

        if self.do_live_analysis:
            log.info("Calibrating image processor, please wait...")
            self.image_processor = ImageProcessor(empty_img, start_img)
            log.info("Calibration completed")
        else:
            log.info('Delaying image processor calibration')


    def _get_img_name(self):
        '''TODO'''
        fullmove_number = self.current_board.fullmove_number
        turn_number     = '0' if self.current_board.turn else '1'

        filename = "board-{:03}-{}.jpg".format(fullmove_number, turn_number)
        return os.path.join(self.game_folder, filename)

    def _process_next_move(self, image_path):
        '''TODO'''
        log.debug('running img processing on "{}"...'.format(image_path))
        self.image_processor.process(image_path)
        new_blindboard = self.image_processor.get_blindboard()
        diff = new_blindboard.diff(self.last_valid_blindboard)
        move = core.diffreader.read(diff)
        log.info("see {} playing: {}".format(self._get_turn_str(), move))

        if not self.current_board.is_legal(move):
            log.warn("Illegal move: {}".format(move))
            raise IllegalMove(move)

        # the move is legit: change state of board and blindboard
        self.current_board.push(move)
        self.last_valid_blindboard = new_blindboard
        self._print_game_status()

        if self.do_show_each_move:
            self.do_show(None)

        return move

    def _print_game_status(self):
        assert self.do_live_analysis

        turn = self._get_turn_str()
        if self.current_board.is_check():
            log.info("{} is in check!".format(turn))
        elif self.current_board.is_checkmate():
            log.info("{} is checkmated. Game over!".format(turn))
        elif self.current_board.is_stalemate():
            log.info("{} to move is in stalemate. Game over!".format(turn))

    def _get_turn_str(self):
        return "White" if self.current_board.turn else "Black"

