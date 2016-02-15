'''Logging functions'''

import sys
import colorama
import logging
import inspect


do_show_debug_messages = False
'''Set this boolean variable to `True` to enable debug messages'''

def multiline(func_call):
    '''Decorator that splits a multi-line string argument to a function into one
    function call per line'''
    def _f(msg):
        msg = str(msg)
        for line in msg.split('\n'):
            func_call(line)
    return _f

def __getCaller():
    '''Returns a tuple `module_name, module_fullname` of the calling module

    `module_name` is the name of the module without the enclosing package,
    whereas `module_fullname` is prefixed with the package name.
    '''
    stack = inspect.stack()

    # ugly hack, in case we're stacking decorators...
    frame_counter = 3
    while stack[frame_counter][1] == __file__:
        frame_counter += 1

    frame = stack[frame_counter]
    frame_filename = frame[1]
    frame_fullname = frame[0].f_globals['__name__']
    module_name = inspect.getmodulename(frame_filename)

    return module_name, frame_fullname

@multiline
def info(msg):
    '''Print out an information message to stdout'''
    print('{}[ INFO ] ({}):{} {}'.format(colorama.Style.DIM, __getCaller()[1],
        colorama.Style.RESET_ALL, msg))
    return


@multiline
def warn(msg):
    '''Print out a warning message to stderr'''
    print('{}[ WARN ] ({}):{} {}'.format(colorama.Fore.YELLOW, __getCaller()[1],
                                  colorama.Fore.RESET,msg), file=sys.stderr)
    return


def error(msg):
    '''Print out an error message to stderr then kill the program'''
    print('{}[ERROR ] ({}):{} {}'.format(colorama.Fore.RED, __getCaller()[1],
            colorama.Fore.RESET, msg), file=sys.stderr)
    sys.exit(42)


@multiline
def debug(msg):
    '''Print out a debug message to stdout.

    Setting `debug.do_show_debug_messages` to `False` will disable all calls to
    this function.
    '''
    if not do_show_debug_messages:
        return
    print('{}[DEBUG ] ({}):{} {}'.format(colorama.Fore.CYAN, __getCaller()[1],
        colorama.Fore.RESET, msg))


