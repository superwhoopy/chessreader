import os
from sklearn import svm
import numpy as np
from enum import Enum

from imgprocessor import ImageProcessor

class Square:
    OCCUPIED = True
    EMPTY    = False

improc = ImageProcessor(trace=True)
MODULE_FOLDER = os.path.join(os.path.dirname(__file__), "..")
EMPTY_BOARD   = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/empty.jpg')
START_BOARD   = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/start.jpg')
MOVE          = os.path.join(MODULE_FOLDER,
                        'tests/pictures/game001/board-003-1.jpg')

improc.load_empty_board(EMPTY_BOARD)
improc.load_starting_position(START_BOARD)

diff_img = improc._compute_diff_image(improc.starting_pos_img)
img_matrix = improc.cut_squares(diff_img, improc._edges)
classifier = svm.SVC(gamma=.001)

square_occupancy = [ Square.OCCUPIED for _ in range(16) ] + \
                   [ Square.EMPTY    for _ in range(32) ] + \
                   [ Square.OCCUPIED for _ in range(16) ]
all_squares = [ img_matrix[i][j] for i in range(8) for j in range(8) ]

n_samples = len(all_squares)
data = np.reshape(all_squares,(n_samples, -1))

classifier.fit(data, square_occupancy)

loaded_move = improc.load_image(MOVE)
move_img_matrix = improc.cut_squares(loaded_move, improc._edges)

# TODO
