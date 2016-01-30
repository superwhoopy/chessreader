
import math
import operator
import os

import chess
from chess import BLACK, WHITE

import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from skimage import feature, io, color, exposure, transform
from skimage.transform import hough_line, hough_line_peaks
from sklearn.neighbors import KNeighborsClassifier

from chessboard.board import BlindBoard

from skimage.color.adapt_rgb import adapt_rgb, each_channel

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
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class ImageProcessor(object):
    """
    This class is used to process a raw image sent from the camera and transform
    it into a `BlindBoard` object. The main method is `process`.
    The following steps are applied to analyze the image:

    At the beginning of the game (inside `__init__`) :

    * CHESSBOARD LINES DETECTION: we use a Canny edge detector to turn the RGB image of the empty chessboard
    into a binary image where the lines separating the squares of the chessboard are clearly visible.
    Then we use the Hough line detection algorithm and pick the 8 most likely horizontal and vertical
    lines in the image.

    * COLOR CLASSIFIER TRAINING: we train a k-nearest neighbour classifier to detect the black and
    white colors using the picture of the beginning of the game, where the position of each piece is known.
    For this classification, each datapoint is one piece, and each piece is represented by an RGB tuple
    corresponding to the average color within the piece (the region of the piece is determined using
    the same method as in occupancy matrix estimation, cf. next bullet point)

    For each new picture during the game:  (inside `process`)

    * OCCUPANCY MATRIX ESTIMATION: we determine for each square whether it is occupied or not.
    To do this, we take the absolute difference between the current image and the image of the empty chessboard.
    All images are in grayscale for this step, and in order to detect black pieces on black squares,
    we use a gamma correction (with gamma < 1) on the input images and on the difference image.
    Then, we apply a binary thresholding to the image (using the Otsu method) and for each square,
    we say that it is occupied if more than 10% of its pixels are white after thresholding.

    * BLINDBOARD ESTIMATION: for each occupied square, we use the color classifier to determine the
    color of the corresponding piece.

    """

    LINE_ANGLE_TOLERANCE = 0.1  # tolerance threshold (in radians) to filter horizontal and vertical lines
    OCCUPANCY_THRESHOLD = 0.2  # if a binary square image has more than this proportion of white pixels, it is considered occupied
    TEMP_DIR = "temp_images"  # name of directory where intermediary images will be stored

    def __init__(self, empty_chessboard_image_path, start_image_path, verbose=False):
        """
        Class constructor. Should be called only once at the start of the game.
        Requires the image of the empty chessboard and the image of the chessboard with all pieces in initial position.
        Computes chessboard edges (costly), and builds training data for color classification using the start image.

        Args:
            empty_chessboard_image_path: path to the image of the empty chessboard.
            start_image_path: path to the image of the chessboard with all pieces in initial position.
            verbose: if `True`, logging messages will be displayed and intermediary images
                will be saved in the current directory inside the `temp_images` folder.
        """

        self.empty_chessboard_image = self.resize_image(io.imread(empty_chessboard_image_path))
        self.initial_image = self.resize_image(io.imread(start_image_path))
        self.image = None
        self.temp_image_dir = None
        self._hough_lines = None
        self._edges = None
        self._square_images = None
        self._processed_square_images = None
        self._occupancy_matrix = None
        self._blindboard_matrix = None
        self.verbose = verbose
        self.color_classifier = None
        self.diff_image = None

        if verbose:
            if not os.path.isdir(self.TEMP_DIR):
                os.mkdir(self.TEMP_DIR)
            print("Computing chessboard edges...")

        self._edges, self._hough_lines = self.find_edges(self.empty_chessboard_image)

        if verbose:
            self.plot_chessboard_with_edges(self.empty_chessboard_image)
            print("Training color classifier...")

        self.train_color_classifier()

    def process(self, image_path, reference=None):
        """
        Run the image analysis pipeline to compute the blindboard.

        Args:
            image_path: path to image to analyze
            reference: (optional) a `numpy.array` storing the "actual" blindboard. If provided,
                accuracy metrics will be displayed.
        """

        if self.verbose:
            self.temp_image_dir = os.path.join(self.TEMP_DIR,
                                               os.path.basename(image_path).rsplit(".", 1)[0])
            if not os.path.isdir(self.temp_image_dir):
                os.makedirs(self.temp_image_dir)
            print("Processing `{0}`...".format(os.path.basename(image_path)))

        self.image = self.resize_image(io.imread(image_path))

        if self.verbose:
            self.save_image("image.png", self.image)
            print("Computing occupancy matrix...")

        self._occupancy_matrix = self.compute_occupancy_matrix()

        if reference is not None:
            print("Occupancy matrix estimation")
            false_positives = np.sum(self._occupancy_matrix & ~reference)
            false_negatives = np.sum(~self._occupancy_matrix & reference)
            print("False positives:", false_positives, "- False negatives:", false_negatives)

        if self.verbose:
            print("Computing blindboard matrix...")
        self._blindboard_matrix = self.compute_blindboard_matrix()

        if reference is not None:
            number_errors = int(np.sum(self._blindboard_matrix != reference))
            print("Number of mislabeled pieces: {0}".format(number_errors))

        if self.verbose:
            print("Blindboard matrix:")
            print(self._blindboard_matrix)

    @staticmethod
    def compute_canny_image(image):
        '''apply a canny filter to each channel of an RGB image and combine
        the results using a logical OR'''
        image_gray = exposure.equalize_hist(color.rgb2gray(image))
        image_canny = np.zeros_like(image_gray)
        for i in range(3):
            normalized_image = image[:, :, i]
            image_canny = np.logical_or(image_canny, feature.canny(normalized_image, 1))
        return image_canny

    def find_edges(self, image):
        """
        Takes as input an RGB chessboard image and computes the Hough lines
        and intersections between them corresponding to the edges of the 64
        squares. The function returns a tuple whose first elsment is a
        numpy array with shape (N,2) where N is the number of
        intersections between the Hough lines. The second element is a list
        of tuples (intensity, abs(theta), abs(r)) describing the significant
        Hough lines (in polar coordinates).
        """

        image_canny = self.compute_canny_image(image)
        h, theta, d = hough_line(image_canny)
        min_distance = int(math.floor(image.shape[1] / 11))  # TODO better way?
        hough_peaks = hough_line_peaks(h, theta, d, min_distance=min_distance, threshold=40)  # TODO adjust

        vertical_lines = []
        horizontal_lines = []
        intersections = []

        for intensity, theta, r in zip(*hough_peaks):
            if abs(theta) < self.LINE_ANGLE_TOLERANCE:
                vertical_lines.append((intensity, theta, r))
            elif abs(abs(theta) - math.pi / 2) < self.LINE_ANGLE_TOLERANCE:
                horizontal_lines.append((intensity, abs(theta), abs(r)))

        # only keep the 9 most significant lines of each direction
        # and sort them by radius, to return edges from top to bottom, and left to right
        for lines in (horizontal_lines, vertical_lines):
            lines.sort(reverse=True)
            del lines[9:]
            lines.sort(key=operator.itemgetter(2))

        for _, theta1, r1 in horizontal_lines:
            for __, theta2, r2 in vertical_lines:
                # can this be numerically unstable? denum is below 1e-10 in some cases here
                denum = np.cos(theta1) * np.sin(theta2) - np.sin(theta1) * np.cos(theta2)
                x_inter = (r1 * np.sin(theta2) - r2 * np.sin(theta1)) / denum
                y_inter = (r2 * np.cos(theta1) - r1 * np.cos(theta2)) / denum
                if 0 < x_inter < image.shape[1] and 0 < y_inter < image.shape[0]:
                    intersections.append((x_inter, y_inter))

        hough_lines = horizontal_lines + vertical_lines
        # TODO we don't check that we get 9x9 intersections...
        return np.array(intersections), hough_lines

    def train_color_classifier(self):
        """
        Train the color classifier using the initial image,
        for which we know the position of the pieces. We isolate an image of each of the 32
        pieces, and we represent each of them by an RGB tuple corresponding to the average
        color within a small disk centered on the middle of each square image.
        We then train the classifier using these data as features and the pieces' labels.

        In verbose mode, a PCA plot of the training data is saved in `temp_images`.
        """

        self.color_classifier = KNeighborsClassifier(metric=color.deltaE_ciede94)
        initial_squares = self.cut_squares(color.rgb2lab(self.initial_image), self._edges)
        training_data = np.zeros((32, 3))  # 32 pieces (16 black, 16 white), 3 color channels
        training_labels = [BLACK for _ in range(16)] + [WHITE for _ in range(16)]
        pieces_indices = [(i, j) for i in [0, 1, 6, 7] for j in range(8)]

        binary_diff_squares = self.cut_squares(self.compute_binary_diff_image(self.initial_image), self._edges)

        for k, index in enumerate(pieces_indices):
            square = initial_squares[index]
            mask = binary_diff_squares[index]
            training_data[k, :] = np.mean(square[mask], axis=0)

        if self.verbose:
            self.save_pca_plot(training_data, training_labels, self.TEMP_DIR)

        self.color_classifier.fit(training_data, training_labels)

    @staticmethod
    def cut_squares(input_image, edges):
        """
        Takes as input a chessboard image and uses the estimated square edges `self._edges` to
        cut the image into 64 square images.
        Args:
            input_image: image to cut
            edges: `numpy.array` with edges coordinates

        Returns: an 8x8 matrix where each entry is a square image (with the same dtype as `input_image`)
        """
        images_matrix = np.empty((8, 8), dtype=object)
        edges_matrix = np.reshape(edges, (9, 9, 2))
        for i in range(8):
            for j in range(8):
                top_left = edges_matrix[i][j]
                bottom_right = edges_matrix[i + 1][j + 1]
                square_image = input_image[math.floor(top_left[1]):math.ceil(bottom_right[1]),
                               math.floor(top_left[0]):math.ceil(bottom_right[0])]
                images_matrix[i, j] = square_image
        return images_matrix

    def compute_occupancy_matrix(self):

        occupancy_matrix = np.empty((8, 8), dtype=bool)
        occupancy_matrix.fill(False)

        binary_diff_image = self.compute_binary_diff_image(self.image)
        binary_diff_squares = self.cut_squares(binary_diff_image, self._edges)

        if self.verbose:
            self.plot_square_images(binary_diff_squares,
                                    os.path.join(self.temp_image_dir,
                                    "diff_bw_squares.png"))

        self._processed_square_images = np.empty((8, 8), dtype=np.ndarray)

        for i in range(binary_diff_squares.shape[0]):
            for j in range(binary_diff_squares.shape[1]):
                square = binary_diff_squares[i, j]
                n_pixels = square.shape[0] * square.shape[1]
                # TODO improve this rule ?
                occupancy_matrix[i, j] = np.sum(square) / n_pixels > self.OCCUPANCY_THRESHOLD
                self._processed_square_images[i, j] = square

        return occupancy_matrix

    def compute_binary_diff_image(self, new_image):
        """
        Compute an Otsu-thresholded image corresponding to the
        absolute difference between the empty chessboard image and the
        current image.
        """
        adj_start_image = exposure.adjust_gamma(color.rgb2gray(self.empty_chessboard_image), 0.1)
        # gamma values have a strong impact on classification
        adj_image = exposure.adjust_gamma(color.rgb2gray(new_image), 0.1)
        diff_image = exposure.adjust_gamma(np.abs(adj_image - adj_start_image), 0.3)
        self.diff_image = diff_image
        return diff_image > threshold_otsu(diff_image)

    def compute_blindboard_matrix(self):

        square_images = self.cut_squares(color.rgb2lab(self.image), self._edges)
        square_colors = np.zeros((64, 3))

        if self.verbose:
            # an 8x8 matrix which will store the representative color of each
            # square (as 20x20 pixels images)
            square_color_images = np.zeros((8,8,20,20,3), dtype=np.uint8)

        for k, ((i,j), square_image) in enumerate(np.ndenumerate(square_images)):
            if self._occupancy_matrix[i,j]:
                piece_mask = self._processed_square_images[i,j]
                square_colors[k,] = np.percentile(square_image[piece_mask], 60, axis=0)
                if self.verbose:
                    rgb_col = color.lab2rgb(np.array(square_colors[k,], ndmin=3))[0,0] * 255.
                    square_color_images[i,j,:,:,] = np.array(rgb_col, dtype=np.uint8)
            elif self.verbose:
                square_color_images[i,j,:,:,] = np.array([0,0,255], dtype=np.uint8)

        if self.verbose:
            self.plot_square_images(square_color_images,
                    os.path.join(self.temp_image_dir, "square_colors.png"))

        occupied_squares = np.reshape(self._occupancy_matrix, (64,))
        square_colors = square_colors[occupied_squares]

        predictions = self.color_classifier.predict(square_colors)
        if self.verbose:
            self.save_pca_plot(square_colors, predictions, self.temp_image_dir)

        estimates = np.empty((64,), dtype=object)
        # `None` == empty square, `True` = white piece, `False` = black piece
        estimates.fill(None)
        estimates[occupied_squares] = predictions
        return np.reshape(estimates, (8, 8))

    @staticmethod
    def resize_image(img, width=500):
        '''resize an image to have a given width, preserving the h/w ratio'''
        x2 = width
        x1,y1 = img.shape[:2]
        return transform.resize(img, (x2, y1*x2/x1))

    def get_blindboard(self):
        """
        Converts the numpy array `_blindboard_matrix` into a BlindBoard object
        """
        if self._blindboard_matrix is None:
            raise ImageProcessorException("The `.process` method has not been called on this object yet")
        occupied_squares = {}
        for (i,j), entry in np.ndenumerate(self._blindboard_matrix):
            file = j ; rank = 7-i
            if entry is not None:
                occupied_squares[chess.square(file, rank)] = bool(entry)
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

    def plot_chessboard_with_edges(self, chessboard_image):
        plt.imshow(chessboard_image, cmap=plt.cm.gray)
        n_rows, n_cols = chessboard_image.shape[:2]

        for _, angle, dist in self._hough_lines:
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - n_cols * np.cos(angle)) / np.sin(angle)
            plt.plot((0, n_cols), (y0, y1), '-r')

        for i, edge in enumerate(self._edges, 1):
            plt.text(edge[0], edge[1], str(i), color='blue')

        plt.axis((0, n_cols, n_rows, 0))
        plt.savefig(os.path.join(self.TEMP_DIR, "edges_and_squares.png"))

    @staticmethod
    def plot_square_images(matrix, file_path):
        fig, axes = plt.subplots(nrows=8, ncols=8)
        for i in range(8):
            for j in range(8):
                axes[i][j].imshow(matrix[i, j], cmap=plt.cm.gray)
                axes[i][j].axis('off')
        plt.plot()
        plt.savefig(file_path)

    def save_image(self, name, image, path=None):
        if path is None:
            path = self.temp_image_dir
        plt.imsave(os.path.join(path, name), image, cmap=plt.cm.gray)


if __name__ == "__main__":

    # FOR DEBUGGING PURPOSES

    @adapt_rgb(each_channel)
    def rgb_rescale_intensity(img):
        p2, p98 = np.percentile(img, (10, 90))
        return exposure.rescale_intensity(img, in_range=(p2, p98))

    def plot(img):
        plt.imshow(img, cmap=plt.cm.gray)

    os.chdir(os.path.split(__file__)[0])
    print("TEST")

    pcr = ImageProcessor("../tests/pictures/board-0.jpg",
                         "../tests/pictures/board-1.jpg", verbose=True)
    pcr.process('../tests/pictures/board-3.jpg')
    print(pcr.get_blindboard())



