import utils

import subprocess
import queue
import threading

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


class StreamReader(threading.Thread):
    def __init__(self, stream, queue):
        self.stream = stream
        self.queue = queue

    def run(self):
        while True:
            line = self.stream.readline()
            self.queue.put(line)


class IOProcess:

    def __init__(self, cmdline):
        try:
            self.proc = subprocess.Popen( cmdline,
                            universal_newlines = True,
                            stdin = subprocess.PIPE,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE)
        except (subprocess.SubprocessError, OSError) :
            utils.log.error('unable to fork {}'.format(cmdline))

        # launch a thread that reads from stdout of the process
        self._read_queue = queue.Queue()
        self.reader      = StreamReader(self.proc.stdout, self._read_queue)

    def read(self):
        # TODO: return list of lines
        pass

    def writeline(self, line):
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

