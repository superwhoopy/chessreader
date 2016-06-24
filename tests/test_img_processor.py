import os
# import nose

from utils.log import debug, warn
from imgprocessor import ImageProcessor
import tests.utils


def compare_blindboards(expected, actual, file_name=None):
    if not expected == actual:
        info = ""
        if file_name:
            info = "for `%s`" % os.path.basename(file_name)
        warn("BlindBoards are different %s: expected" % info)
        warn(expected)
        warn("But found instead:")
        warn(actual)
        raise AssertionError("BlindBoards are different")
    return True


PGN_FILENAME = 'game.pgn'
def process_game(dirpath):
    pgn_file = os.path.join(dirpath, PGN_FILENAME)
    expected_blind_boards = \
        list(tests.utils.boards_from_pgn(pgn_file, use_blindboards=True))
    images = tests.utils.collect_test_images(dirpath)

    debug("Calibrating image processor...")
    processor = ImageProcessor(images[0], images[1])

    for img, expected_board in zip(images[1:], expected_blind_boards):
        debug("Processing `{}`...".format(os.path.basename(img)))
        processor.process(img)
        processed_board = processor.get_blindboard()
        yield compare_blindboards, expected_board, processed_board, img



# @nose.tools.nottest
def test_imgage_processor():
    '''Test image processor (multiple tests)'''
    # retrieve all the images paths and sort
    dirs = ( os.path.join('tests/pictures/', game) \
                for game in ['game000', 'game001'] )

    # utils.log.do_show_debug_messages = True
    for dirpath in dirs:
        yield from process_game(dirpath)

