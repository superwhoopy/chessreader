''' TODO
'''

import nose
import sys

from . import utils

def run():
    nose.main(argv=[sys.argv[0]], exit=False)

if __name__ == '__main__':
    run()
