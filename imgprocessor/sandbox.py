import os
from sklearn import svm
import numpy as np
from enum import Enum

from imgprocessor import ImageProcessor

class Square:
    OCCUPIED = True
    EMPTY    = False

def all_pics_same_size(matrix, min_width=None, min_height=None):
    min_width = min_width or \
            min(matrix[i][j].shape[0] for i in range(8) for j in range(8))
    min_height = min_height or \
            min(matrix[i][j].shape[1] for i in range(8) for j in range(8))

    for i in range(8):
        for j in range(8):
            img = matrix[i][j]
            w_start = (img.shape[0] - min_width)/2
            w_end   = w_start + min_width
            h_start = (img.shape[1] - min_height)/2
            h_end   = h_start + min_height
            matrix[i][j] = matrix[i][j][w_start:w_end, h_start:h_end]
            assert matrix[i][j].shape[0] == min_width
            assert matrix[i][j].shape[1] == min_height

    return matrix


improc = ImageProcessor(trace=False)
MODULE_FOLDER = os.path.join(os.path.dirname(__file__), "..")
EMPTY_BOARD   = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/empty.jpg')
START_BOARD   = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/start.jpg')
MOVE          = os.path.join(MODULE_FOLDER,
                        'tests/pictures/game001/board-003-1.jpg')

improc.load_empty_board(EMPTY_BOARD)
improc.load_starting_position(START_BOARD)

diff_img = improc._compute_diff_image(improc.starting_pos_img)
img_matrix = improc.cut_squares(diff_img, improc._edges)

cropped_matrix = all_pics_same_size(img_matrix)
samples_shape = cropped_matrix[0][0].shape

classifier = svm.SVC(gamma=.001)

square_occupancy = [ Square.OCCUPIED for _ in range(16) ] + \
                   [ Square.EMPTY    for _ in range(32) ] + \
                   [ Square.OCCUPIED for _ in range(16) ]
all_squares = [ img_matrix[i][j] for i in range(8) for j in range(8) ]

n_samples = len(all_squares)
training_data = np.reshape(all_squares,(n_samples, -1))

classifier.fit(training_data, square_occupancy)

loaded_move = improc.load_image(MOVE)
diff_img = improc._compute_diff_image(loaded_move)
move_img_matrix = improc.cut_squares(diff_img, improc._edges)
move_img_matrix = all_pics_same_size(move_img_matrix, samples_shape[0],
        samples_shape[1])
data = [move_img_matrix[i][j] for i in range(8) for j in range(8)]
data = np.reshape(data, (n_samples, -1))
prediction = classifier.predict(data)

for i in range(8):
    new_line = ""
    for j in range(8):
        new_line += " {:1}".format(prediction[i*8 + j])
    print(new_line)
