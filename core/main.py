from enum import Enum
import chess
import cmd
import os

from imgprocessor import ImageProcessor
import capture, utils
from utils import log
import core.diffreader
from core.diffreader import IllegalMove

from chessboard import BlindBoard



class PlayMode(Enum):
    '''TODO'''
    ONE_PLAYER = 1
    TWO_PLAYERS = 2


class Core(cmd.Cmd):
    '''TO-FUCKING-DO'''

    # TODO: by default, cleanup directory mess when this object is destructed
    # (use __enter__() and __exit__() methods?)

    intro  = 'chessreader game shell'
    prompt = ' (chessreader-PLAYING) '

    def __init__(self, capture_engine=None):
        self.capture_engine        = capture_engine or capture.Fswebcam()
        self.image_processor       = None # TODO
        self.chess_engine          = None  # TODO
        self.game_folder           = utils.create_new_game_folder()
        self.last_valid_blindboard = BlindBoard.get_starting_board()
        self.current_board         = chess.Board()

    # cmd-related stuff ########################################################
    def emptyline(self):
        pass

    def preloop(self):
        '''TODO'''

        # TODO: this is really messy - we need to handle calibration failure!
        empty_img = 'empty.png'
        start_img = self._get_img_name()

        log.info("Prepare the empty chessboard and press Enter...")
        input()
        self.capture_engine.capture(empty_img)
        log.info("Prepare the chessboard in starting position "
                 "and press Enter...")
        input()
        self.capture_engine.capture(start_img)

        log.info("Calibrating image processor, please wait...")
        self.image_processor = ImageProcessor(empty_img, start_img)

        log.info("Calibration finished. The game is on!")

    ############################################################################

    def _get_img_name(self):
        '''TODO'''
        turn = '0' if self.current_board.turn else '1'
        filename = "board-{:03}-{}".format(
                self.current_board.fullmove_number, turn)
        return os.path.join(self.game_folder, filename)

    def process_next_move(self):
        image_path = self.capture_engine.capture()
        self.image_processor.process(image_path)
        new_blindboard = self.image_processor.get_blindboard()
        diff = new_blindboard.diff(self.last_valid_blindboard)
        move = core.diffreader.read(diff)

        utils.log.info("{} played: {}".format(self.get_turn_str(), move))

        if not self.current_board.is_legal(move):
            utils.log.warn("Illegal move: {}".format(move))
            raise IllegalMove(move)

        # the move is legit: change state of board and blindboard
        self.current_board.push(move)
        self.last_valid_blindboard = new_blindboard
        return move, self.check_game_status()

    def check_game_status(self):
        turn = self.get_turn_str()

        if self.current_board.is_check():
            utils.log.info("{} is in check!".format(turn))
        elif self.current_board.is_checkmate():
            utils.log.info("{} is checkmated. Game over!".format(turn))
            return False
        elif self.current_board.is_stalemate():
            utils.log.info("{} to move is in stalemate. Game over!".
                    format(turn))
            return False

        return True

    def kill(self):
        pass

    def get_turn_str(self):
        return "White" if self.current_board.turn else "Black"


