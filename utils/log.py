'''Logging functions'''

import sys
import colorama


def info(msg):
    '''Print out an information message to stdout'''
    print('[INFO]  {}'.format(msg))
    return


def warn(msg):
    '''Print out a warning message to stderr'''
    print('{}[WARN]{}  {}'.format(colorama.Fore.YELLOW,
                                  colorama.Fore.RESET,msg), file=sys.stderr)
    return


def error(msg):
    '''Print out an error message to stderr then kill the program'''
    print('{}[ERROR]{} {}'.format(colorama.Fore.RED, colorama.Fore.RESET, msg),
            file=sys.stderr)
    sys.exit(42)


def debug(msg):
    '''Print out a debug message to stdout.

    Setting `debug.do_show_debug_messages` to `False` will disable all calls to
    this function.
    '''
    if not debug.do_show_debug_messages:
        return
    print('{}[DEBUG]{} {}'.format(colorama.Fore.CYAN, colorama.Fore.RESET, msg))

debug.do_show_debug_messages = False
'''Set this boolean variable to `True` to enable debug messages'''

