import os
import subprocess
import queue

import re

import tests
import utils.log as log
from threading import Thread

################################################################################

def singleton_get(singleton_set: set):
    '''Return the single object in a set

    Raises:
        IndexError: if `singleton_set` is not a singleton
    '''
    if len(singleton_set) != 1:
        raise IndexError()
    return next(iter(singleton_set))

################################################################################

# TODO: StreamReader and Proc are not used anymore - drop it?

class StreamReader(Thread):
    '''Threaded Stream Reader

    Runs a daemon in background that continuously tries to read from a stream,
    and puts each line read into a queue. Particularly useful when interacting
    with a subprocess through stdout/stderr.'''

    # automatically kill the thread if it's the only one left alive
    daemon = True

    def __init__(self, stream, read_queue):
        '''Thread default constructor.

        The thread is not launched until its `start()` method is called.

        Args:
            stream: A file object such as `stdout`, `stderr`, ...
            read_queue: Output queue where the lines read from `stream` will
                be pushed to. Final carriage return will be removed.
        '''
        Thread.__init__(self)
        self.stream = stream
        self._queue = read_queue

    def run(self):
        '''Main thread method: continusouly read from the stream'''
        while True:
            line = self.stream.readline()
            # remove the final carriage return
            self._queue.put(line[:-1])


class Proc:
    '''Call an external process and interact with it on standard I/O'''

    # the StreamReader on stdout will push to this queue
    _stdout_queue = queue.Queue()

    def __init__(self, cmdline):
        '''Subprocess constructor

        Launch an interactive subprocess, such as a shell, or any command line
        interface.

        Args:
            cmdline (list of str): Command line and its arguments
        '''
        try:
            # create the process
            self.proc = subprocess.Popen( cmdline,
                            universal_newlines = True,
                            stdin = subprocess.PIPE,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE)
        except (subprocess.SubprocessError, OSError) :
            log.error('unable to fork {}'.format(cmdline))

        # launch a thread that continuously reads from stdout of the process
        self.reader = StreamReader(self.proc.stdout, self._stdout_queue)
        self.reader.start()


    def read_stdout(self):
        '''Read all the lines printed to stdout by the process until now.

        Returns:
            A list of string, each string is a line without the final carriage
            return character.
        '''
        lines = []
        while True:
            try:
                lines.append(self._stdout_queue.get_nowait())
            except queue.Empty:
                return lines

    def readline_stdout(self, block=True):
        '''Read one line from stdout.

        Args:
            block (bool): when `True`, this call will be blocking if nothing new
                was printed to stdout.

        Returns:
            One line of stdout, without the final carriage return, or `None` if
            `block` was set to `False` and no new output was available.
        '''
        try:
            line = self._stdout_queue.get(block=block)
        except queue.Empty:
            line = None
        return line

    def writeline(self, line):
        '''Write a line to stdin, and flush it

        Args:
            line (str): line to write to stdin, without trailing carriage return
        '''
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()

#########################################################################

USER_FOLDER = os.path.join(os.path.expanduser("~"), ".chessreader")
GAME_REGEX  = re.compile("game_[0-9]+")


def get_existing_games():
    result = []
    if not os.path.isdir(USER_FOLDER):
        return result
    for file_name in os.listdir(USER_FOLDER):
        file_path = os.path.join(USER_FOLDER, file_name)
        if os.path.isdir(file_path) and GAME_REGEX.match(file_name):
            result.append(file_path)
    return tests.utils.natural_sort(result)


def create_new_game_folder():
    if not os.path.isdir(USER_FOLDER):
        os.mkdir(USER_FOLDER)
    games = get_existing_games()
    new_dir = os.path.join(USER_FOLDER, "game_{0}".format(len(games)))
    os.mkdir(new_dir)
    return new_dir

