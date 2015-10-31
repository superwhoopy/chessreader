% matplotlib
inline

from __future__ import division
from __future__ import print_function

import math
import operator

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature, io, color, filters, exposure
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import corner_harris, corner_subpix, corner_peaks

from sklearn.cluster import KMeans


class ImageProcessor(object):
    ANGLE_TOL = 0.1

    def __init__(self, image_path):
        self.image = io.imread(image_path)
        self.n_rows, self.n_cols = self.image.shape[:2]

        self._hough_lines = None
        self._edges = None
        self._square_images = None
        self._bw_square_images = None
        self._occupancy_matrix = None

    def process(self, show_details=False, reference=None):
        self.find_edges()

        if show_details:
            self.plot_chessboard_with_edges()

        self.cut_squares()
        self.compute_occupancy_matrix()

        if show_details:
            self.plot_bw_square_images()

        if reference is not None:
            print("OCCUPANCY MATRIX ESTIMATION")
            false_positives = np.sum(self._occupancy_matrix & ~reference)
            false_negatives = np.sum(~self._occupancy_matrix & reference)
            print("False positives:", false_positives, "- False negatives:", false_negatives)

        self.compute_blindboard_matrix()

        if reference is not None:  # TODO currently we treat black and white symmetrically in the error computation
            print("BLINDBOARD ESTIMATION")
            number_errors = int(min(np.sum(self._blindboard_matrix != reference),
                                    np.sum(self._blindboard_matrix != ~reference)))
            print("Number of mislabeled pieces:", number_errors)

        if show_details:
            print(self._blindboard_matrix)


    def compute_canny_image(self):
        image_gray = exposure.equalize_hist(color.rgb2gray(self.image))
        image_canny = np.zeros_like(image_gray)
        for i in range(3):
            normalized_image = self.image[:, :, i]
            image_canny = image_canny | feature.canny(normalized_image, 1)
        return image_canny

    def find_edges(self):
        image_canny = self.compute_canny_image()
        h, theta, d = hough_line(image_canny)
        min_distance = int(math.floor(self.image.shape[0] / 11))  # TODO better way?
        hough_peaks = hough_line_peaks(h, theta, d, threshold=60, num_peaks=25,
                                       min_distance=min_distance)  # TODO adjust

        lines = {'horizontals': [], 'verticals': []}
        intersections = []

        for intensity, theta, r in zip(*hough_peaks):
            if abs(theta) < self.ANGLE_TOL:
                lines['verticals'].append((intensity, theta, r))
            elif abs(abs(theta) - math.pi / 2) < self.ANGLE_TOL:
                lines['horizontals'].append((intensity, abs(theta), abs(r)))

        # sort lines by radius, to return edges from top to bottom, and left to right
        for key in lines:
            # lines[key].sort()
            # lines[key] = lines[key][0:9]
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
        self._edges = np.array(intersections)

    def cut_squares(self):
        images_matrix = np.empty((8, 8), dtype=object)
        edges_matrix = np.reshape(self._edges, (9, 9, 2))
        for i in range(8):
            for j in range(8):
                top_left = edges_matrix[i][j]
                bottom_right = edges_matrix[i + 1][j + 1]
                square_image = self.image[math.floor(top_left[1]):math.ceil(bottom_right[1]),
                               math.floor(top_left[0]):math.ceil(bottom_right[0])]
                images_matrix[i, j] = square_image
        self._square_images = images_matrix

    def plot_chessboard_with_edges(self):
        plt.imshow(self.image, cmap=plt.cm.gray)

        for _, angle, dist in self._hough_lines:
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - self.n_cols * np.cos(angle)) / np.sin(angle)
            plt.plot((0, self.n_cols), (y0, y1), '-r')

        for i, edge in enumerate(self._edges, 1):
            plt.text(edge[0], edge[1], str(i), color='blue')

        plt.axis((0, self.n_cols, self.n_rows, 0))
        plt.title('Hough lines and edges')
        plt.plot()

    def plot_bw_square_images(self):
        fig, axes = plt.subplots(nrows=8, ncols=8)
        for i in range(8):
            for j in range(8):
                axes[i][j].imshow(self._bw_square_images[i, j], cmap=plt.cm.gray)
                axes[i][j].axis('off')
        plt.plot()
        plt.title('Pieces contour detection')

    @staticmethod
    def is_square_occupied(square_image):
        img_out = np.zeros_like(square_image[:, :, 0])
        for i in range(3):
            img_canny = feature.canny(square_image[:, :, i], 0.5)
            img_out = img_out | img_canny
        img_out = ndi.binary_fill_holes(img_out)
        return np.mean(img_out) > 0.13, img_out

    def compute_occupancy_matrix(self):
        self._occupancy_matrix = np.zeros((8, 8))
        self._bw_square_images = np.empty((8, 8), dtype=np.ndarray)

        for i in range(self._square_images.shape[0]):
            for j in range(self._square_images.shape[1]):
                result = self.is_square_occupied(self._square_images[i, j])
                self._occupancy_matrix[i, j], self._bw_square_images[i, j] = result

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
        square_colors = self.apply_to_matrix(self.get_representative_color, self._square_images, 3)
        square_colors = np.reshape(square_colors, (64, 3))
        occupied_squares = np.reshape(self._occupancy_matrix, (64,))
        square_colors = square_colors[occupied_squares]

        kmeans_predictions = KMeans(n_clusters=2, random_state=0).fit_predict(square_colors)
        # use -1 and 1 as labels
        kmeans_predictions[kmeans_predictions == 0] = -1
        estimates = np.zeros((64,))
        estimates[occupied_squares] = kmeans_predictions
        self._blindboard_matrix = np.reshape(estimates, (8, 8))
