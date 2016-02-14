import glob
import os

import skimage.io
import skimage.color
import skimage.exposure

import numpy as np
import matplotlib.pyplot as plt

# load files
from skimage import img_as_float

MODULE_FOLDER = os.path.dirname(__file__)
train_empty = glob.glob(os.path.join(MODULE_FOLDER, "training", "empty-substract", "*.jpg"))
train_filled = glob.glob(os.path.join(MODULE_FOLDER, "training", "filled-substract", "*.jpg"))

test = glob.glob(os.path.join(MODULE_FOLDER, "testing", "*.jpg"))

# os.makedirs("train_histograms/filled")
# os.makedirs("train_histograms/empty")


'''
BW histograms are very similar for non-diff images.
But, for diffed images, histograms for empty images are spikey (chi-square/gaussian?)
with a single spike, whereas histograms for filled images are bimodal-ish, with one
large peak and another smaller one.
'''

for d in ("empty", "filled"):
    if not os.path.isdir("train_histograms/%s" % d):
        os.makedirs("train_histograms/%s" % d)

os.makedirs("test_histograms", exist_ok=True)

for img_path in test:
    img = skimage.color.rgb2gray(skimage.io.imread(img_path))
    img = img_as_float(skimage.color.rgb2gray(img))
    # img = skimage.exposure.adjust_gamma(img, 0.1)
    print ("Processing {0}".format(img_path))
    _, histogram, __ = plt.hist(np.ndarray.flatten(img), bins=100, range=(0,1),
                                facecolor='green', normed=True)
    name = os.path.basename(img_path).split(".")[0]
    plt.savefig("test_histograms/%s" % ("hist-%s.png" % name))
    plt.close()

# for k,imgs in enumerate((train_empty, train_filled)):
#     for i,train in enumerate(imgs):
#         if i > 4:
#             break
#         img = skimage.io.imread(train, as_grey=True)/255.
#         print ("Processing {0}".format(train))
#         _, histogram, __ = plt.hist(np.ndarray.flatten(img), bins=100, range=(0,1),
#                                     facecolor='green', normed=True)
#         plt.savefig("train_histograms/%s/hist-%d.png" %
#                     ("empty" if k==0 else "filled", i))

filled_path = 'testing-diff/square-11-t.jpg'
empty_path = 'testing-diff/square-12-t.jpg'

empty = img_as_float(skimage.io.imread(empty_path, as_grey=True))
filled = img_as_float(skimage.io.imread(filled_path, as_grey=True))

def show(img):
    plt.imshow(img, cmap=plt.cm.gray)

empty_obs = np.ndarray.flatten(empty)
filled_obs = np.ndarray.flatten(filled)
