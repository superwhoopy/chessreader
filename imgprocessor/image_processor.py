# Default python lib ###########################################################

import math
from math import floor, ceil
import operator
import os

# Additional packages ##########################################################

import chess
from chess import BLACK, WHITE

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from skimage import feature, io, color, exposure, transform
from skimage.transform import hough_line, hough_line_peaks
from sklearn.neighbors import KNeighborsClassifier
from skimage.color.adapt_rgb import adapt_rgb, each_channel

# Internal packages ############################################################

from chessboard.board import BlindBoard
from utils.log import error, debug, info, warn

################################################################################

"""
TODO (30/01/2016)
* if the gamma correction is too low on the absolute diff image,
    we may fail to detect some pieces.
* we might try to improve the difference between black and white pieces

Currently, for classification, we encode the colors in lab space and use the
deltaE_cie76 color distance which is supposed to be a more accurate reflection
of `color differences` as percieved by the human eye (as opposed to RGB
euclidian distance which fails to capture the `non-linearity` of human color
perception).
http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.deltaE_cie76


"""


class ImageProcessorException(Exception):
    '''Default exception raised on various occasions by `ImageProcessor`'''
    pass


class ImageProcessor(object):
    """
    This class is used to process a raw image sent from the camera and transform
    it into a `BlindBoard` object. The main method is `process`.
    The following steps are applied to analyze the image:

    At the beginning of the game (inside `__init__`) :

    * CHESSBOARD LINES DETECTION: we use a Canny edge detector to turn the RGB
      image of the empty chessboard into a binary image where the lines
      separating the squares of the chessboard are clearly visible. Then we
      use the Hough line detection algorithm and pick the 8 most likely
      horizontal and vertical lines in the image.

    * COLOR CLASSIFIER TRAINING: we train a k-nearest neighbour classifier to
      detect the black and white colors using the picture of the beginning of
      the game, where the position of each piece is known. For this
      classification, each datapoint is one piece, and each piece is
      represented by an RGB tuple corresponding to the average color within the
      piece (the region of the piece is determined using the same method as in
      occupancy matrix estimation, cf. next bullet point)

    For each new picture during the game: (inside `process`)

    * OCCUPANCY MATRIX ESTIMATION: we determine for each square whether it is
      occupied or not. To do this, we take the absolute difference between the
      current image and the image of the empty chessboard. All images are in
      grayscale for this step, and in order to detect black pieces on black
      squares, we use a gamma correction (with gamma < 1) on the input images
      and on the difference image. Then, we apply a binary thresholding to the
      image (using the Otsu method) and for each square, we say that it is
      occupied if more than 10% of its pixels are white after thresholding.

    * BLINDBOARD ESTIMATION: for each occupied square, we use the color
      classifier to determine the color of the corresponding piece.

    Attributes:
        _edges (nparray): Array of coordinates of the limits of each square of
            the chessboard, i.e. the intersection of horizontal & vertical
            lines.

        _hough_lines (array): Array of 9 horizontal lines followed by 9 vertical
            lines, as returned by the `hough_line()` method.

        empty_chessboard_image (ndarray): image of the empty chessboard (i.e. no
            pieces on it), used for edge detection

        TODO
    """

    LINE_ANGLE_TOLERANCE = 30./180. * math.pi
    '''tolerance threshold (in radians) to filter horizontal and vertical
    lines'''

    ANGLE_EPSILON = 1e-3
    '''tolerance (in radians) to consider two angles to be close enough for the
    corresponding Hough lines to be parallel'''

    TRACE_DIR = "trace_images"
    '''name of directory where intermediary images will be stored'''

    def __init__(self, empty_board_path=None, starting_board_path=None,
            trace=False, verbose=False):
        """
        Class constructor. Should be called only once at the start of the game.
        Requires the image of the empty chessboard and the image of the
        chessboard with all pieces in initial position.
        Computes chessboard edges (costly), and builds training data for color
        classification using the start image.

        Args:
            TODO
        """

        self.verbose                  = verbose
        self.trace                    = trace

        self.image                    = None
        self.temp_image_dir           = None
        self._hough_lines             = None

        self._edges                   = None
        self._processed_square_images = None
        self._occupancy_matrix        = None
        self._blindboard_matrix       = None

        self.color_classifier         = None

        # mkdir the trace directory if it does not exist
        if self.trace:
            if not os.path.isdir(self.TRACE_DIR):
                os.mkdir(self.TRACE_DIR)

        if empty_board_path:
            self.load_empty_board(empty_board_path)
        if starting_board_path:
            self.load_starting_position(starting_board_path)



    def load_empty_board(self, image_path):
        '''TODO'''
        self.empty_chessboard_image = self.load_image(image_path)
        debug("Computing chessboard edges...")
        self._edges, self._hough_lines = \
                self._find_edges(self.empty_chessboard_image)

        if self.trace:
            self.plot_chessboard_with_edges(self.empty_chessboard_image)



    def load_starting_position(self, image_path):
        '''TODO'''
        self.starting_pos_img = self.load_image(image_path)

        debug("Training color classifier...")
        self._train_color_classifier(self.starting_pos_img)

        self.calibrate_occupancy_threshold()
        debug("Occupancy threshold set to {}".format(self.occupancy_threshold))



    def process(self, image_path, reference=None):
        """
        Run the image analysis pipeline to compute the blindboard.

        Args:
            image_path: path to image to analyze
            reference: (optional) a `numpy.array` storing the "actual"
                blindboard. If provided, accuracy metrics will be displayed.
        """

        if self.trace:
            self.temp_image_dir = \
                os.path.join(self.TRACE_DIR,
                        os.path.basename(image_path).rsplit(".", 1)[0])
            if not os.path.isdir(self.temp_image_dir):
                os.makedirs(self.temp_image_dir)

        debug("Processing `{}`...".format(os.path.basename(image_path)))
        self.image = self.load_image(image_path)

        if self.trace:
            self.save_image("image.png", self.image)

        debug("Computing occupancy matrix...")
        self._occupancy_matrix = self.compute_occupancy_matrix()
        debug(str(self._occupancy_matrix))
        debug("Computing blindboard matrix...")
        self._blindboard_matrix = self.compute_blindboard_matrix()

        if reference is not None:
            info("Occupancy matrix estimation")
            false_positives = np.sum(self._occupancy_matrix & ~reference)
            false_negatives = np.sum(~self._occupancy_matrix & reference)
            info("False positives: {} - False negatives: {}".format(
                    false_positives, false_negatives))

            number_errors = int(np.sum(self._blindboard_matrix != reference))
            info("Number of mislabeled pieces: {}".format(number_errors))

        debug("Blindboard matrix:")
        debug(str(self._blindboard_matrix))


    @staticmethod
    def compute_canny_image(image):
        '''apply a canny filter to each channel of an RGB image and combine
        the results using a logical OR'''
        image_gray = exposure.equalize_hist(color.rgb2gray(image))
        image_canny = np.zeros_like(image_gray)
        for i in range(3):
            normalized_image = image[:, :, i]
            image_canny = \
                np.logical_or(image_canny, feature.canny(normalized_image, 1))
        return image_canny


    def _find_edges(self, image):
        """
        Takes as input an RGB chessboard image and computes the Hough lines
        and intersections between them corresponding to the edges of the 64
        squares. The function returns a tuple whose first element is a
        numpy array with shape (N,2) where N is the number of
        intersections between the Hough lines. The second element is a list
        of tuples (intensity, abs(theta), abs(r)) describing the significant
        Hough lines (in polar coordinates).
        """

        image_canny = self.compute_canny_image(image)
        h, theta, d = hough_line(image_canny)

        '''
        Small trick: we expect to get something like 11 horizontal lines and
        another 11 vertical lines: indeed the chessboard is made of 8 rows and
        8 columns, i.e. 9 lines, plus 2 lines for the edges of the chessboard.
        So, to help hough_line_peaks() let's set the min. distance expected
        between two lines to the size of the image divided by 11.
        TODO: better way? Might cause problems if the board does not fit the
        whole frame...
        '''
        min_distance = int(floor(image.shape[1] / 11))

        hough_peaks = hough_line_peaks(h, theta, d, min_distance=min_distance,
                                       threshold=40)  # TODO adjust threshold

        # separate horizontal from vertical lines
        vertical_lines = []
        horizontal_lines = []

        '''
        In Hough space, straight lines are parametrized by coefficients (r,θ):
        r = x.cos(θ) + y.sin(θ) ; (r,θ) are the polar coordinates of
        the point on the line which lies closest to the origin, with r being
        either positive or negative, and -π/2 ≤ θ ≤ π/2   (with π/2 ≈ 1.57)

        A horizontal line is therefore one with θ=±π/2, and a vertical line is
        one with θ=0.
        Also, two lines are parallel iif they have the same θ (mod π/2).

        Finally, in scikit-image, keep in mind that the orientation is :

                                O----->
                                |     (x axis)
                                |
                                V (y axis)

        (where the origin is the top-left corner of our image)
        And as a consequence, positive angles are counted from the X axis toward
        the Y axis (i.e. "clockwise").
        '''

        for intensity, theta, r in zip(*hough_peaks):
            if np.fabs(theta) < self.LINE_ANGLE_TOLERANCE:
                vertical_lines.append((intensity, theta, r))
            elif np.fabs(np.fabs(theta) - math.pi / 2) < self.LINE_ANGLE_TOLERANCE:
                horizontal_lines.append((intensity, theta, r))

        '''
        Only keep the 9 most significant lines of each direction and sort
        them by absolute radius, to return edges from top to bottom,
        and left to right. We sort by *absolute* radius because horizontal
        lines can be parametrized either as 'θ=π/2, r>0' or 'θ=-π/2, r<0'
        '''
        for i,lines in enumerate((horizontal_lines, vertical_lines)):
            lines.sort(reverse=True)   # reverse-sort by intensity
            del lines[9:]              # only keep the 9 first
            lines.sort(key=lambda entry: np.fabs(entry[2]))  # sort by absolute radius

        '''
        Now let's find the intersections of these 18 lines.
        We are solving the following system:
        r1 = x.cos(θ1) + y.sin(θ1)  and  r2 = x.cos(θ2) + y.sin(θ2)   for (x,y)
        The non-degenerate solution is:
        x = (r1.sin(θ2) - r2.sin(θ1))/D  and  y = (r2.cos(θ1) - r1.cos(θ2))/D
        with D = cos(θ1).sin(θ2) - sin(θ1).cos(θ2) = sin(θ2-θ1)
        We can see that this holds only if θ1 ≠ θ2, i.e. only if the lines are
        not parallel (which makes sense!).
        '''
        intersections = []
        for _, theta1, r1 in horizontal_lines:
            for _, theta2, r2 in vertical_lines:
                # don't attempt to compute the intersection if the lines are parallel
                denum = np.sin(theta2 - theta1)
                if denum < self.ANGLE_EPSILON:
                    continue
                x_inter = (r1 * np.sin(theta2) - r2 * np.sin(theta1)) / denum
                y_inter = (r2 * np.cos(theta1) - r1 * np.cos(theta2)) / denum
                # register this intersection iff. it is *inside* the image
                if 0 < x_inter < image.shape[1] and 0 < y_inter < image.shape[0]:
                    intersections.append((x_inter, y_inter))

        hough_lines = horizontal_lines + vertical_lines
        # TODO we don't check that we get 9x9 intersections...

        '''
        Since the horizontal and vertical lines have been sorted by increasing
        value of |r|, the intersections are now sorted from top-left to bottom-right
        corner of the image.
        '''
        return np.array(intersections), hough_lines


    def _train_color_classifier(self, img):
        """
        Train the color classifier using the initial image, for which we know
        the position of the pieces. We isolate an image of each of the 32
        pieces, and we represent each of them by an RGB tuple corresponding to
        the average color within a small disk centered on the middle of each
        square image.  We then train the classifier using these data as
        features and the pieces' labels.

        In verbose mode, a PCA plot of the training data is saved in
        `temp_images`.
        """

        self.color_classifier = \
            KNeighborsClassifier(metric=color.deltaE_ciede94)
        if self._edges is None:
            raise ImageProcessorException(
                    'cannot train the classifier prior to edge detection')

        initial_squares = self.cut_squares(color.rgb2lab(img), self._edges)

        # 32 pieces (16 black, 16 white), 3 color channels
        training_data = np.zeros((32, 3))
        training_labels = [BLACK for _ in range(16)] + \
                          [WHITE for _ in range(16)]
        pieces_indices = [(i, j) for i in [0, 1, 6, 7] for j in range(8)]

        binary_diff_squares = self.cut_squares(
                self.compute_binary_diff_image(img), self._edges)

        # TODO: seems suspicious

        for k, index in enumerate(pieces_indices):
            square = initial_squares[index]
            mask = binary_diff_squares[index]
            training_data[k, :] = np.mean(square[mask], axis=0)

        if self.trace:
            self.save_pca_plot(training_data, training_labels, self.TRACE_DIR)

        self.color_classifier.fit(training_data, training_labels)


    @staticmethod
    def cut_squares(input_image, edges):
        """
        Takes as input a chessboard image and uses the estimated square edges
        `self._edges` to cut the image into 64 square images.

        Args:
            input_image: image to cut
            edges: `numpy.array` with edges coordinates

        Returns:
            an 8x8 matrix where each entry is a square image (with the same
            dtype as `input_image`). The entry at [0,0] corresponds to the
            top-left square on the chessboard.
        """
        images_matrix = np.empty((8, 8), dtype=object)
        edges_matrix = np.reshape(edges, (9, 9, 2))

        for i in range(8):
            for j in range(8):
                ''' Bear in mind that:
                            0-----------> (x)
                            | * (top-left)
                            |
                            |      * (bottom-right)
                            |
                            V (y)
                '''
                top_left = edges_matrix[i][j]
                bottom_right = edges_matrix[i + 1][j + 1]
                x_top_left, y_top_left = map(lambda n: int(round(n)), top_left)
                x_bottom_right, y_bottom_right = map(lambda n: int(round(n)), bottom_right)
                assert x_top_left < x_bottom_right
                assert y_top_left < y_bottom_right
                square_image = input_image[ y_top_left:y_bottom_right,
                                            x_top_left:x_bottom_right ]
                images_matrix[i, j] = square_image
        return images_matrix


    def calibrate_occupancy_threshold(self):
        binary_diff_image = \
                self.compute_binary_diff_image(self.starting_pos_img)
        binary_diff_squares = self.cut_squares(binary_diff_image, self._edges)

        threshold = 1.
        # for each cut square
        for i in [0, 1, 6, 7]:
            for j in range(8):
                square = binary_diff_squares[i, j]
                ratio = np.sum(square) / square.size
                threshold = min(threshold, ratio)

        self.occupancy_threshold = threshold - .05


    def compute_occupancy_matrix(self, img=None):
        img = img or self.image

        occupancy_matrix = np.empty((8, 8), dtype=bool)
        occupancy_matrix.fill(False)

        binary_diff_image = self.compute_binary_diff_image(img)
        binary_diff_squares = self.cut_squares(binary_diff_image, self._edges)

        if self.trace:
            self.plot_square_images(binary_diff_squares,
                                    os.path.join(self.temp_image_dir,
                                    "diff_bw_squares.png"))

        self._processed_square_images = np.empty((8, 8), dtype=np.ndarray)

        for i in range(binary_diff_squares.shape[0]):
            for j in range(binary_diff_squares.shape[1]):
                square = binary_diff_squares[i, j]
                n_pixels = square.size
                occupancy_matrix[i, j] = \
                    np.sum(square) / n_pixels > self.occupancy_threshold
                self._processed_square_images[i, j] = square

        return occupancy_matrix


    def compute_binary_diff_image(self, new_image, binary=True):
        """
        Compute an Otsu-thresholded image corresponding to the
        absolute difference between the empty chessboard image and the
        current image.
        """
        empty = color.rgb2gray(self.empty_chessboard_image)
        new = color.rgb2gray(new_image)

        new1 = exposure.adjust_sigmoid(new)
        empty1 = exposure.adjust_sigmoid(empty)
        diff1 = np.fabs(new1 - empty1)

        new2 = exposure.adjust_sigmoid(new, cutoff=0.0001)
        empty2 = exposure.adjust_sigmoid(empty, cutoff=0.0001)
        diff2 = np.fabs(new2 - empty2)

        diff = np.fmax(exposure.rescale_intensity(diff1),
                       exposure.rescale_intensity(diff2))

        diff = exposure.adjust_sigmoid(diff, cutoff=0.1)

        return diff > threshold_otsu(diff) if binary else diff


    def compute_blindboard_matrix(self):

        square_images = self.cut_squares(color.rgb2lab(self.image), self._edges)
        square_colors = np.zeros((64, 3))

        if self.trace:
            # an 8x8 matrix which will store the representative color of each
            # square (as 20x20 pixels images)
            square_color_images = np.zeros((8,8,20,20,3), dtype=np.uint8)

        for k, ((i,j), square_image) in \
                enumerate(np.ndenumerate(square_images)):
            if self._occupancy_matrix[i,j]:
                piece_mask = self._processed_square_images[i,j]
                square_colors[k,] = \
                        np.percentile(square_image[piece_mask], 60, axis=0)
                if self.trace:
                    rgb_col = 255. * \
                            color.lab2rgb(np.array(square_colors[k,],
                                ndmin=3))[0,0]
                    square_color_images[i,j,:,:,] = np.array(rgb_col,
                                                             dtype=np.uint8)
            elif self.trace:
                square_color_images[i,j,:,:,] = np.array([0,0,255],
                                                         dtype=np.uint8)

        if self.trace:
            self.plot_square_images(square_color_images,
                    os.path.join(self.temp_image_dir, "square_colors.png"))

        occupied_squares = np.reshape(self._occupancy_matrix, (64,))
        square_colors = square_colors[occupied_squares]

        predictions = self.color_classifier.predict(square_colors)
        if self.trace:
            self.save_pca_plot(square_colors, predictions, self.temp_image_dir)

        estimates = np.empty((64,), dtype=object)
        # `None` == empty square, `True` = white piece, `False` = black piece
        estimates.fill(None)
        estimates[occupied_squares] = predictions
        return np.reshape(estimates, (8, 8))


    @staticmethod
    def load_image(img_path, resize=True, resize_width=500):
        '''TODO'''
        img = skimage.img_as_float(io.imread(img_path))

        if resize:
            x2    = resize_width
            x1,y1 = img.shape[:2]
            img = transform.resize(img, (x2, y1*x2/x1))

        return img


    def get_blindboard(self):
        """
        Converts the numpy array `_blindboard_matrix` into a BlindBoard object
        """
        if self._blindboard_matrix is None:
            raise ImageProcessorException(
                "The `.process` method has not been called on this object yet")
        occupied_squares = {}
        for (i, j), entry in np.ndenumerate(self._blindboard_matrix):
            col = j ; row = 7-i
            if entry is not None:
                occupied_squares[chess.square(col, row)] = bool(entry)

        return BlindBoard.from_dict(occupied_squares)

    # ------------------------ PLOTTING METHODS ------------------------

    @staticmethod
    def save_pca_plot(X, labels, basedir):
        pca = PCA(n_components=2)
        X_r = pca.fit_transform(X)
        plt.clf()
        colors = ["brown" if k == BLACK else "beige" for k in labels]
        plt.scatter(X_r[:, 0], X_r[:, 1], color=colors, edgecolors="black")
        plt.savefig(os.path.join(basedir, "colors_pca.png"))
        plt.close()

    def plot_chessboard_with_edges(self, chessboard_image):
        plt.imshow(chessboard_image, cmap=plt.cm.gray)
        n_rows, n_cols = chessboard_image.shape[:2]

        for _, angle, dist in self._hough_lines:
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - n_cols * np.cos(angle)) / np.sin(angle)
            plt.plot((0, n_cols), (y0, y1), '-r')

        for i, edge in enumerate(self._edges, 1):
            plt.text(edge[0], edge[1], str(i), color='blue')
            plt.plot(edge[0], edge[1], 'b+')

        plt.axis((0, n_cols, n_rows, 0))
        plt.savefig(os.path.join(self.TRACE_DIR, "edges_and_squares.png"))
        plt.close()

    @staticmethod
    def plot_square_images(matrix, file_path):
        fig, axes = plt.subplots(nrows=8, ncols=8)
        for i in range(8):
            for j in range(8):
                axes[i][j].imshow(matrix[i, j], cmap=plt.cm.gray)
                axes[i][j].axis('off')
        plt.plot()
        plt.savefig(file_path)
        plt.close()

    @staticmethod
    def _plot_all_lines(img, peaks, imgname="all_lines.png"):
        # peaks: list of tuples (_, angle, dist)
        plt.clf()
        axis = plt.imshow(img)
        rows, cols, _ = img.shape
        for _, angle, dist in peaks:
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
            plt.plot((0, cols), (y0, y1), '-r')

        plt.axis((0, cols, rows, 0))
        plt.savefig(os.path.join(ImageProcessor.TRACE_DIR, imgname))
        plt.close()

    def save_image(self, name, image, path=None):
        if path is None:
            path = self.temp_image_dir
        plt.imsave(os.path.join(path, name), image, cmap=plt.cm.gray)
        plt.close()


def show(img):
    plt.imshow(img, cmap=plt.cm.gray)

def image_max(img1, img2):
    return np.fmax(exposure.rescale_intensity(img1), exposure.rescale_intensity(img2))

def apply_to_squares(input_image, edges, func):

    output_image = np.copy(input_image)
    edges_matrix = np.reshape(edges, (9, 9, 2))

    for i in range(8):
        for j in range(8):
            ''' Bear in mind that:
                        0-----------> (x)
                        | * (top-left)
                        |
                        |      * (bottom-right)
                        |
                        V (y)
            '''
            top_left = edges_matrix[i][j]
            bottom_right = edges_matrix[i + 1][j + 1]
            x_top_left, y_top_left = map(lambda n: int(round(n)), top_left)
            x_bottom_right, y_bottom_right = map(lambda n: int(round(n)), bottom_right)
            assert x_top_left < x_bottom_right
            assert y_top_left < y_bottom_right

            square_image = func(input_image[ y_top_left:y_bottom_right,
                                        x_top_left:x_bottom_right ])
            output_image[ y_top_left:y_bottom_right, x_top_left:x_bottom_right ] = \
                square_image

    return output_image

if __name__ == "__main__":
    import utils
    utils.log.do_show_debug_messages = True
    MODULE_FOLDER = os.path.join(os.path.dirname(__file__), "..")
    EMPTY_BOARD = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/empty.jpg')
    START_BOARD= os.path.join(MODULE_FOLDER, 'tests/pictures/game001/start.jpg')
    MOVE = os.path.join(MODULE_FOLDER, 'tests/pictures/game001/board-003-1.jpg')
    imgproc = ImageProcessor(EMPTY_BOARD, START_BOARD, trace=True)
    imgproc.process(MOVE)
    debug(imgproc.get_blindboard())



