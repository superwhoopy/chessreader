from __future__ import division

import math
import operator
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, io, color, filters, exposure
from skimage.transform import hough_line, hough_line_peaks, resize
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans


class ImageProcessor(object):
    """
    This class is used to process a raw image sent from the camera and transform
    it into a `BlindBoard` object. The main method is `process`.
    The following steps are applied to analyze the image:

    1. CHESSBOARD LINES DETECTION: we use a Canny edge detector to turn the RGB image
    into a binary image where the lines separating the squares of the chessboard are clearly visible.
    Then we use the Hough line detection algorithm and pick the 8 most likely horizontal and vertical
    lines in the image.

    2. SQUARES ISOLATION: from the estimated equations of the chessboard lines, we derive analytically
    the intersections of the 16 chessboard lines, and produce a matrix of 8x8 images `self._square_images`
    where each image corresponds to a single square in the chessboard.

    3. OCCUPANCY MATRIX ESTIMATION: we then determine for each square whether it is occupied or not.
    To do this, we take the absolute difference between the current image and the first image of the game
    (where all pieces are in their starting position).
    All images are in grayscale for this step, and in order to detect black pieces on black squares,
    we use a gamma correction (with gamma < 1) on the input images and on the difference image.
    Then, we apply a binary thresholding to the image (using the Otsu method) and for each square,
    we say that it is occupied if more than 10% of the pixels are white.

    4. BLINDBOARD ESTIMATION: for each occupied square, we pick a representative color (average RGB color
    in a disk centered on the midpoint of the square). We then use k-means clustering (k=2) on these data
    (each datapoint represents a piece, and the dimensions are the three RGB channels) to classify each
    piece as 'black' or 'white'.


    """
    LINE_ANGLE_TOLERANCE = 0.1  # tolerance threshold (in radians) to filter horizontal and vertical lines
    TEMP_DIR = "temp_images"  # name of directory where intermediary images will be stored

    def __init__(self, start_image_path):
        """
        Class constructor. Should be called only once at the start of the game.
        Args:
            start_image_path: path to the image where all the pieces are in their starting position.
        """

        self.start_image = io.imread(start_image_path)
        self.image = None
        self.n_rows, self.n_cols = None, None

        self._hough_lines = None
        self._edges = None
        self._square_images = None
        self._processed_square_images = None
        self._occupancy_matrix = None
        self._blindboard_matrix = None
        self.show_details = None

    def process(self, image_path, show_details=False, reference=None):
        """
        Run the image analysis pipeline to compute the blindboard.

        Args:
            image_path: path to image to analyze
            show_details: if `True`, logging messages will be displayed and intermediary images
                will be saved in the current directory inside the `temp_images` folder.
            reference: (optional) a `numpy.array` storing the "actual" blindboard. If provided,
                accuracy metrics will be displayed.

        """

        self.show_details = show_details
        if show_details:
            print("Processing `{0}`...".format(os.path.basename(image_path)))
            if not os.path.isdir(self.TEMP_DIR):
                os.mkdir(self.TEMP_DIR)


        self.image = io.imread(image_path)
        self.n_rows, self.n_cols = self.image.shape[:2]

        if show_details:
            self.save_image("image.jpg", self.image)
            print("Computing edges...")

        self._edges = self.find_edges()

        if show_details:
            self.plot_chessboard_with_edges()
            print("Computing occupancy matrix...")

        self.compute_occupancy_matrix()

        # if show_details:
        #     self.plot_square_images(self._processed_square_images)

        if reference is not None:
            print("OCCUPANCY MATRIX ESTIMATION")
            false_positives = np.sum(self._occupancy_matrix & ~reference)
            false_negatives = np.sum(~self._occupancy_matrix & reference)
            print("False positives:", false_positives, "- False negatives:", false_negatives)

        if show_details:
            print("Computing blindboard matrix...")
        self._blindboard_matrix = self.compute_blindboard_matrix()

        if reference is not None:  # TODO currently we treat black and white symmetrically in the error computation
            print("BLINDBOARD ESTIMATION")
            number_errors = int(min(np.sum(self._blindboard_matrix != reference),
                                    np.sum(self._blindboard_matrix != ~reference)))
            print("Number of mislabeled pieces:", number_errors)

        if show_details:
            print("Blindboard matrix:")
            print(self._blindboard_matrix)

    def compute_canny_image(self):
        image_gray = exposure.equalize_hist(color.rgb2gray(self.image))
        image_canny = np.zeros_like(image_gray)
        for i in range(3):
            normalized_image = self.image[:, :, i]
            image_canny = np.logical_or(image_canny, feature.canny(normalized_image, 1))
        return image_canny

    def find_edges(self):
        image_canny = self.compute_canny_image()
        h, theta, d = hough_line(image_canny)
        min_distance = int(math.floor(self.image.shape[1] / 11))  # TODO better way?
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
                if 0 < x_inter < self.n_cols and 0 < y_inter < self.n_rows:
                    intersections.append((x_inter, y_inter))

        self._hough_lines = lines['horizontals'] + lines['verticals']
        # TODO we don't check that we get 9x9 intersections...
        return np.array(intersections)

    def cut_squares(self, input_image):
        """
        Takes as input a chessboard image and uses the estimated square edges `self._edges` to
        cut the image into 64 square images.
        Args:
            input_image: image to cut

        Returns: an 8x8 matrix where each entry is a square image (with the same dtype as `input_image`)
        """
        images_matrix = np.empty((8, 8), dtype=object)
        edges_matrix = np.reshape(self._edges, (9, 9, 2))
        for i in range(8):
            for j in range(8):
                top_left = edges_matrix[i][j]
                bottom_right = edges_matrix[i + 1][j + 1]
                square_image = input_image[math.floor(top_left[1]):math.ceil(bottom_right[1]),
                                          math.floor(top_left[0]):math.ceil(bottom_right[0])]
                images_matrix[i, j] = square_image
        return images_matrix

    def plot_chessboard_with_edges(self):
        # fig = plt.figure()
        plt.imshow(self.image, cmap=plt.cm.gray)

        for _, angle, dist in self._hough_lines:
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - self.n_cols * np.cos(angle)) / np.sin(angle)
            plt.plot((0, self.n_cols), (y0, y1), '-r')

        for i, edge in enumerate(self._edges, 1):
            plt.text(edge[0], edge[1], str(i), color='blue')

        plt.axis((0, self.n_cols, self.n_rows, 0))
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

    def save_image(self, name, image):
        plt.imsave(os.path.join(self.TEMP_DIR, name), image, cmap=plt.cm.gray)

    def compute_occupancy_matrix(self):
        self._occupancy_matrix = np.empty((8, 8), dtype=bool)
        self._occupancy_matrix.fill(False)

        # compute difference between starting and current image
        adj_start_image = exposure.adjust_gamma(color.rgb2gray(self.start_image), 0.1)
        adj_image = exposure.adjust_gamma(color.rgb2gray(self.image), 0.1)
        diff_image = exposure.adjust_gamma(np.abs(adj_image - adj_start_image), 0.5)
        binary_diff_image = diff_image > threshold_otsu(diff_image)

        # FIXME use a different correction to detect white pieces
        if self.show_details:
            self.save_image("gamma_adj_image.jpg", adj_image)
            self.save_image("diff_gray.jpg", diff_image)
            self.save_image("diff_bw.jpg", binary_diff_image)

        binary_diff_squares = self.cut_squares(binary_diff_image)

        self._processed_square_images = np.empty((8, 8), dtype=np.ndarray)

        for i in range(binary_diff_squares.shape[0]):
            for j in range(binary_diff_squares.shape[1]):
                square = binary_diff_squares[i,j]
                n_pixels = square.shape[0] * square.shape[1]
                self._occupancy_matrix[i,j] = np.sum(square)/n_pixels > 0.1
                # pdb.set_trace()
                self._processed_square_images[i,j] = square

    @staticmethod
    def get_representative_color(img):
        x_center = math.floor(img.shape[0] / 2)
        y_center = math.floor(img.shape[1] / 2)
        return img[x_center, y_center]

    @staticmethod
    def apply_to_matrix(f, M, odim):
        output = np.empty(M.shape + (odim,))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                output[i, j] = f(M[i, j])
        return output

    def compute_blindboard_matrix(self):
        square_images = self.cut_squares(self.image)
        square_colors = self.apply_to_matrix(self.get_representative_color, square_images, 3)
        square_colors = np.reshape(square_colors, (64, 3))
        occupied_squares = np.reshape(self._occupancy_matrix, (64,))
        square_colors = square_colors[occupied_squares]

        kmeans_predictions = KMeans(n_clusters=2, random_state=0).fit_predict(square_colors)
        # use -1 and 1 as labels
        kmeans_predictions[kmeans_predictions == 0] = -1
        estimates = np.zeros((64,))
        estimates[occupied_squares] = kmeans_predictions
        return np.reshape(estimates, (8, 8))


if __name__ == "__main__":
    IMAGES_FOLDER = '/Users/daniel/Desktop/chess/chess-pictures-other'
    base_img = os.path.join(IMAGES_FOLDER, 'Picture 13.jpg')
    test_files = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER)
                  if f.endswith(".jpg") and f != "Picture 13.jpg"]
    processor = ImageProcessor(base_img)
    processor.process(random.choice(test_files), show_details=True)