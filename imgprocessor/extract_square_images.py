import itertools
import os
import glob
import skimage.io

from . import ImageProcessor
import utils

utils.log.do_show_debug_messages = True

local_dir = os.path.dirname(__file__)
pictures_fd = os.path.join(local_dir, "..", "tests", "pictures")
empties = glob.glob(os.path.join(pictures_fd, "*", "empty.jpg"))
starts = glob.glob(os.path.join(pictures_fd, "*", "start.jpg"))

empty_dir = os.path.join(local_dir, "training", "empty-substract")
filled_dir = os.path.join(local_dir, "training", "filled-substract")

for d in (empty_dir, filled_dir):
    if not os.path.isdir(d):
        os.makedirs(d)

# os.makedirs(os.path.join(local_dir, "testing"))

MODULE_FOLDER = os.path.join(os.path.dirname(__file__), "..")
EMPTY_BOARD = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/empty.jpg')
START_BOARD= os.path.join(MODULE_FOLDER, 'tests/pictures/game001/start.jpg')
MOVE = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/board-003-1.jpg')
proc = ImageProcessor(EMPTY_BOARD, START_BOARD, trace=True)
squares = proc.cut_squares(proc.compute_binary_diff_image(
    proc.load_image(MOVE), binary=False), proc._edges)
for k,(i,j) in enumerate(itertools.product(range(8), range(8))):
    skimage.io.imsave(os.path.join(local_dir, "testing-diff", "square-%d-t.jpg" % k),
                      squares[i,j])

def make_testing_diff():
    for empty, start in zip(empties, starts):
        proc = ImageProcessor(empty, start, trace=True)
        print(empty, start)

        start_squares = proc.cut_squares(proc.compute_binary_diff_image(
            proc.starting_pos_img, binary=False), proc._edges)

        for k,(i,j) in enumerate(itertools.product(range(8), range(8))):
            save_dir = filled_dir if i in {0,1,6,7} else empty_dir
            skimage.io.imsave(os.path.join(
                save_dir, "square-%d-f.jpg" % k), start_squares[i][j])

make_testing_diff()

def make_testing_images():
    MODULE_FOLDER = os.path.join(os.path.dirname(__file__), "..")
    EMPTY_BOARD = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/empty.jpg')
    START_BOARD= os.path.join(MODULE_FOLDER, 'tests/pictures/game001/start.jpg')
    MOVE = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/board-003-1.jpg')
    proc = ImageProcessor(EMPTY_BOARD, START_BOARD, trace=True)
    squares = proc.cut_squares(proc.load_image(MOVE), proc._edges)
    for k,(i,j) in enumerate(itertools.product(range(8), range(8))):
        skimage.io.imsave(os.path.join(local_dir, "testing", "square-%d-t.jpg" % k),
                          squares[i,j])


def make_training_images():
    for empty, start in zip(empties, starts):
        proc = ImageProcessor(empty, start, trace=True)
        print(empty, start)

        empty_squares = proc.cut_squares(proc.empty_chessboard_image, proc._edges)
        start_squares = proc.cut_squares(proc.starting_pos_img, proc._edges)
        for k,(i,j) in enumerate(itertools.product(range(8), range(8))):
            skimage.io.imsave(os.path.join(
                empty_dir, "square-%d.jpg" % k), empty_squares[i][j])
            save_dir = filled_dir if i in {0,1,6,7} else empty_dir
            skimage.io.imsave(os.path.join(
                save_dir, "square-%d-f.jpg" % k), start_squares[i][j])

'''
show(np.fabs(exposure.adjust_sigmoid(new)-exposure.adjust_sigmoid(empty)))
gives amazing results except for black on black

new1 = exposure.adjust_sigmoid(new)
empty1 = exposure.adjust_sigmoid(empty)
diff1 = np.fabs(new1 - empty1)

new2 = exposure.adjust_sigmoid(new, cutoff=0.0001)
empty2 = exposure.adjust_sigmoid(empty, cutoff=0.0001)
diff2 = np.fabs(new2 - empty2)

diff = np.fmax(exposure.rescale_intensity(diff1), exposure.rescale_intensity(diff2))
exposure.adjust_sigmoid(diff, cutoff=0.1)
'''
