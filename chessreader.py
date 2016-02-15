'''chessreader init module '''

import argparse
import sys
import colorama

import tests, utils, core

################################################################################

ARG_MAIN_HELP = ''

ARG_TEST_HELP = \
    'Launch the test suite of Chessreader; all other arguments are ignored'

ARG_DEBUG_HELP = \
    'Display debugging messages to stdout'


PARSED_ARGS = None

def parse_arguments():
    '''Parse command-line arguments and return the parsed output'''
    parser = argparse.ArgumentParser(description=ARG_MAIN_HELP)

    parser.add_argument('--test', help=ARG_TEST_HELP, action="store_true")
    parser.add_argument('-d', '--debug', help=ARG_DEBUG_HELP,
                        action="store_true")

    return parser.parse_args()

################################################################################


def main():
    '''Launch function'''
    colorama.init()

    utils.log.do_show_debug_messages = PARSED_ARGS.debug

    # Launch test suite?
    if PARSED_ARGS.test:
        log.info("running chessreader test base")
        tests.run()
        sys.exit(0)

    # default: call the shell
    core.Shell().cmdloop()

    sys.exit(0)


if __name__ == '__main__':
    PARSED_ARGS = parse_arguments()
    main()

