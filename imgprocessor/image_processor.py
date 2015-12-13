from __future__ import division, print_function
import math
import operator
import os
import random
from aetypes import Enum
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import feature, io, color, exposure
from skimage.transform import hough_line, hough_line_peaks
from skimage.filters import threshold_otsu
from sklearn.neighbors import KNeighborsClassifier

"""
TODO (13/12/2015)
* The whole thing works generally well, color detection is fine,
but sometimes occupancy matrix has one or two errors.
This might be due to an issue in the way the squares are isolated? or maybe the thresholding?
* write a method to return an actual `BlindBoard` object based on the `self._blindboard_matrix`
* write proper testing on a set of images
"""


class ImageProcessor(object):
    """
    This class is used to process a raw image sent from the camera and transform
    it into a `BlindBoard` object. The main method is `process`.
    The following steps are applied to analyze the image:

    At the beginning of the game:

    * CHESSBOARD LINES DETECTION: we use a Canny edge detector to turn the RGB image
    into a binary image where the lines separating the squares of the chessboard are clearly visible.
    Then we use the Hough line detection algorithm and pick the 8 most likely horizontal and vertical
    lines in the image.

    * COLOR CLASSIFIER TRAINING: we train a k-nearest neighbour classifier to detect the black and
    white colors using the picture of the beginning of the game, where the position of each piece is known.
    For this classification, each datapoint is one piece, and each piece is represented by an RGB tuple
    corresponding to the average color within a small disk centered on the image of the piece.

    For each new picture during the game:

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
    TEMP_DIR = "temp_images"  # name of directory where intermediary images will be stored

    # this enum is used for the classification labels
    class Color(Enum):
        BLACK = -1
        WHITE = 1

    def __init__(self, blank_image_path, start_image_path, verbose=False):
        """
        Class constructor. Should be called only once at the start of the game.
        Requires the image of the empty chessboard and the image of the chessboard with all pieces in initial position.
        Computes chessboard edges (costly), and builds training data for color classification using the start image.

        Args:
            blank_image_path: path to the image of the empty chessboard.
            start_image_path: path to the image of the chessboard with all pieces in initial position.
            verbose: if `True`, logging messages will be displayed and intermediary images
                will be saved in the current directory inside the `temp_images` folder.
        """

        self.empty_chessboard_image = io.imread(blank_image_path)
        self.initial_image = io.imread(start_image_path)
        self.image = None
        self.n_rows, self.n_cols = None, None
        self.temp_image_dir = None
        self._hough_lines = None
        self._edges = None
        self._square_images = None
        self._processed_square_images = None
        self._occupancy_matrix = None
        self._blindboard_matrix = None
        self.verbose = verbose
        self.color_classifier = None

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

        self.image = io.imread(image_path)
        self.n_rows, self.n_cols = self.image.shape[:2]  # TODO remove this

        if self.verbose:
            self.save_image("image.jpg", self.image)
            print("Computing occupancy matrix...")

        self.compute_occupancy_matrix()

        if reference is not None:
            print("Occupancy matrix estimation")
            false_positives = np.sum(self._occupancy_matrix & ~reference)
            false_negatives = np.sum(~self._occupancy_matrix & reference)
            print("False positives:", false_positives, "- False negatives:", false_negatives)

        if self.verbose:
            print("Computing blindboard matrix...")
        self._blindboard_matrix = self.compute_blindboard_matrix()

        if reference is not None:  # TODO currently we treat black and white symmetrically in the error computation
            number_errors = int(np.sum(self._blindboard_matrix != reference))
            print("Number of mislabeled pieces: {0}".format(number_errors))

        if self.verbose:
            print("Blindboard matrix:")
            print(self._blindboard_matrix)

    @staticmethod
    def compute_canny_image(image):
        image_gray = exposure.equalize_hist(color.rgb2gray(image))
        image_canny = np.zeros_like(image_gray)
        for i in range(3):
            normalized_image = image[:, :, i]
            image_canny = np.logical_or(image_canny, feature.canny(normalized_image, 1))
        return image_canny

    def find_edges(self, image):
        image_canny = self.compute_canny_image(image)
        h, theta, d = hough_line(image_canny)
        min_distance = int(math.floor(image.shape[1] / 11))  # TODO better way?
        hough_peaks = hough_line_peaks(h, theta, d, min_distance=min_distance, threshold=40)  # TODO adjust

        lines = {'horizontals': [], 'verticals': []}
        intersections = []

        for intensity, theta, r in zip(*hough_peaks):
            if abs(theta) < self.LINE_ANGLE_TOLERANCE:
                lines['verticals'].append((intensity, theta, r))
            elif abs(abs(theta) - math.pi / 2) < self.LINE_ANGLE_TOLERANCE:
                lines['horizontals'].append((intensity, abs(theta), abs(r)))

        # only keep the 9 most significant lines of each direction
        # and sort them by radius, to return edges from top to bottom, and left to right
        for key in lines:
            lines[key].sort(reverse=True)
            lines[key] = lines[key][0:9]
            lines[key].sort(key=operator.itemgetter(2))

        for _, theta1, r1 in lines['horizontals']:
            for __, theta2, r2 in lines['verticals']:
                # can this be numerically unstable? denum is below 1e-10 in some cases here
                denum = np.cos(theta1) * np.sin(theta2) - np.sin(theta1) * np.cos(theta2)
                x_inter = (r1 * np.sin(theta2) - r2 * np.sin(theta1)) / denum
                y_inter = (r2 * np.cos(theta1) - r1 * np.cos(theta2)) / denum
                if 0 < x_inter < image.shape[1] and 0 < y_inter < image.shape[0]:
                    intersections.append((x_inter, y_inter))

        hough_lines = lines['horizontals'] + lines['verticals']
        # TODO we don't check that we get 9x9 intersections...
        return np.array(intersections), hough_lines

    def train_color_classifier(self):
        """
        Train the color classifier using the initial image,
        for which we know the position of the pieces. We isolate an image of each of the 32
        pieces, and we represent each of them by an RGB tuple corresponding to the average
        color within a small disk centered on the middle of each square image.
        We then train the classifier using these data and the pieces labels.

        In verbose mode, a PCA plot of the training data is saved in `temp_images`.
        """

        self.color_classifier = KNeighborsClassifier()
        initial_squares = self.cut_squares(self.initial_image, self._edges)
        training_data = np.zeros((32, 3))  # 32 pieces (16 black, 16 white), 3 color channels
        training_labels = [self.Color.BLACK for _ in range(16)] + [self.Color.WHITE for _ in range(16)]
        pieces_indices = [(i, j) for i in [0, 1, 6, 7] for j in range(8)]

        for k, index in enumerate(pieces_indices):
            square = initial_squares[index]
            training_data[k, :] = self.get_representative_color(square)

        if self.verbose:
            self.save_pca_plot(training_data, training_labels, self.TEMP_DIR)

        self.color_classifier.fit(training_data, training_labels)

    def save_pca_plot(self, X, labels, basedir):
        pca = PCA(n_components=2)
        X_r = pca.fit_transform(X)
        plt.clf()
        colors = ["brown" if k == self.Color.BLACK else "beige" for k in labels]
        plt.scatter(X_r[:, 0], X_r[:, 1], color=colors, edgecolors="black")
        plt.savefig(os.path.join(basedir, "colors_pca.jpg"))

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
        plt.savefig(os.path.join(self.TEMP_DIR, "edges_and_squares.jpg"))

    @staticmethod
    def plot_square_images(matrix):
        fig, axes = plt.subplots(nrows=8, ncols=8)
        for i in range(8):
            for j in range(8):
                axes[i][j].imshow(matrix[i, j], cmap=plt.cm.gray)
                axes[i][j].axis('off')
        plt.plot()
        plt.show()

    def save_image(self, name, image, path=None):
        if path is None:
            path = self.temp_image_dir
        plt.imsave(os.path.join(path, name), image, cmap=plt.cm.gray)

    def compute_occupancy_matrix(self):
        self._occupancy_matrix = np.empty((8, 8), dtype=bool)
        self._occupancy_matrix.fill(False)

        # compute difference between starting and current image
        adj_start_image = exposure.adjust_gamma(color.rgb2gray(self.empty_chessboard_image), 0.1)
        adj_image = exposure.adjust_gamma(color.rgb2gray(self.image), 0.1)
        diff_image = exposure.adjust_gamma(np.abs(adj_image - adj_start_image), 0.5)
        binary_diff_image = diff_image > threshold_otsu(diff_image)

        if self.verbose:
            self.save_image("gamma_adj_image.jpg", adj_image)
            self.save_image("diff_gray.jpg", diff_image)
            self.save_image("diff_bw.jpg", binary_diff_image)

        binary_diff_squares = self.cut_squares(binary_diff_image, self._edges)

        self._processed_square_images = np.empty((8, 8), dtype=np.ndarray)

        for i in range(binary_diff_squares.shape[0]):
            for j in range(binary_diff_squares.shape[1]):
                square = binary_diff_squares[i, j]
                n_pixels = square.shape[0] * square.shape[1]
                # TODO improve this rule ?
                self._occupancy_matrix[i, j] = np.sum(square) / n_pixels > 0.1
                self._processed_square_images[i, j] = square

    @staticmethod
    def get_representative_color(img):
        l_x, l_y = img.shape[:2]
        x, y = np.ogrid[:l_x, :l_y]
        disk_mask = (x - l_x / 2) ** 2 + (y - l_y / 2) ** 2 <= (l_x / 4) ** 2
        return np.mean(img[disk_mask], axis=0)

    @staticmethod
    def apply_to_matrix(f, M, odim):  # TODO remove this method
        output = np.empty(M.shape + (odim,))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                output[i, j] = f(M[i, j])
        return output

    def compute_blindboard_matrix(self):
        square_images = self.cut_squares(self.image, self._edges)
        square_colors = self.apply_to_matrix(self.get_representative_color, square_images, 3)
        square_colors = np.reshape(square_colors, (64, 3))
        occupied_squares = np.reshape(self._occupancy_matrix, (64,))
        square_colors = square_colors[occupied_squares]

        predictions = self.color_classifier.predict(square_colors)
        if self.verbose:
            self.save_pca_plot(square_colors, predictions, self.temp_image_dir)
        estimates = np.zeros((64,))
        estimates[occupied_squares] = predictions
        return np.reshape(estimates, (8, 8))


if __name__ == "__main__":
    IMAGES_FOLDER = '/Users/daniel/Desktop/chess/chess-pictures-other'
    base_img = os.path.join(IMAGES_FOLDER, 'Picture 13.jpg')
    start_img = os.path.join(IMAGES_FOLDER, 'Picture 14.jpg')
    test_files = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER)
                  if f.endswith(".jpg") and f not in {'Picture 13.jpg', 'Picture 14.jpg'}]
    processor = ImageProcessor(base_img, start_img, verbose=True)
    processor.process(random.choice(test_files))
