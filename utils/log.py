'''TODO
'''
import sys
import colorama


def info(msg):
    print('[INFO]  {}'.format(msg))
    return


def warn(msg):
    print('{}[WARN]{}  {}'.format(colorama.Fore.YELLOW,
                                  colorama.Fore.RESET,msg), file=sys.stderr)
    return


def error(msg):
    print('{}[ERROR]{} {}'.format(colorama.Fore.RED, colorama.Fore.RESET, msg),
            file=sys.stderr)
    sys.exit(42)


def debug(msg):
    if not debug.do_show_debug_messages:
        return
    print('{}[DEBUG]{} {}'.format(colorama.Fore.CYAN, colorama.Fore.RESET, msg))

debug.do_show_debug_messages = False

