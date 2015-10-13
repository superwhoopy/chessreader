'''TODO
'''

import argparse
import sys
import colorama

import imgprocessor
import chessboard
import log

################################################################################

# TODO
ARG_MAIN_HELP = ''

ARG_TEST_HELP = \
    'Launch the test suite of Chessreader; all other arguments are ignored'

ARG_DEBUG_HELP = \
    'Display debugging messages to stdout'

def parse_arguments():
    '''TODO
    '''
    parser = argparse.ArgumentParser(description=ARG_MAIN_HELP)

    parser.add_argument('--test', help=ARG_TEST_HELP, action="store_true")
    parser.add_argument('-d', '--debug', help=ARG_DEBUG_HELP,
                        action="store_true")

    return parser.parse_args()

################################################################################

def main(parsed_args):
    '''TODO
    '''
    colorama.init()

    # Enable debug messages?
    log.debug.do_show_debug_messages = parsed_args.debug

    # Launch test suite?
    if parsed_args.test:
        sys.exit(0)

    sys.exit(0)


if __name__ == '__main__':
    PARSED_ARGS = parse_arguments()
    main(PARSED_ARGS)

