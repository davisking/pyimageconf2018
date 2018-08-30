import sys
sys.path = ['./superfast/build'] + sys.path

import numpy as np
from dlib import *
import timeit
import superfast

img = load_rgb_image('images/find_knots/knots3.jpg')


pyr = pyramid_down(2)


img = pyr(pyr(img))

image_window(img)

ig = image_gradients(30)

gimg = as_grayscale(img)
xx = ig.gradient_xx(gimg)
xy = ig.gradient_xy(gimg)
yy = ig.gradient_yy(gimg)

horzvert = find_bright_lines(xx,xy,yy)

blobs = find_bright_keypoints(xx,xy,yy)

lines = suppress_non_maximum_edges(horzvert)


def discard_all_but_largest_blob(img):
    labels, num_blobs = label_connected_blobs(img, connected_if_both_not_zero=True, neighborhood_connectivity=24)
    h = get_histogram(labels, num_blobs)
    # ignore background blobs
    h[0] = 0
    largest_blob = np.argmax(h)
    start = timeit.default_timer()
    #for r in range(img.shape[0]):
    #    for c in range(img.shape[1]):
    #        if labels[r][c] != largest_blob:
    #            img[r][c] = 0
    # you can vectorize this by saying:
    img = img*(labels==largest_blob)
    # But it's still not as fast as just writing it in C++.  (Why?)
    #superfast.zero_pixels_not_labeled_with_val(img, labels, largest_blob)
    stop = timeit.default_timer()
    print("time to mask out small blobs: ", stop-start)
    return img


lines = discard_all_but_largest_blob(lines)

peaks = find_peaks(blobs*lines, non_max_suppression_radius=15)


win2 = image_window(jet(blobs*lines), "blobs*lines")
win3 = image_window(jet(lines), "lines")
win4 = image_window(jet(blobs), "blobs")

win = image_window(img)
win.add_overlay(centered_rects(peaks, 50, 50))

input("hit enter to stop")
