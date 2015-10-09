import time

import skimage.io
import skimage.filter

def say_hello():
    imarray = skimage.io.imread('imgprocessor/samples/flatboard.png', True)
    edge_im = skimage.filter.sobel(imarray)

    skimage.io.imshow(edge_im)

    time.sleep(10)
    print("Hello, World!")

