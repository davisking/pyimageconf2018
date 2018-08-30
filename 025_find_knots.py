import sys
sys.path = ['./superfast/build'] + sys.path

import numpy as np
from dlib import *
import timeit
import superfast

# Works well for this image
img = load_rgb_image('images/find_knots/knots1.jpg')

# but what about this image?
#img = load_rgb_image('images/find_knots/knots3.jpg')

pyr = pyramid_down(2)


img = pyr(pyr(img))

image_window(img)

ig = image_gradients(30)

gimg = as_grayscale(img)
xx = ig.gradient_xx(gimg)
xy = ig.gradient_xy(gimg)
yy = ig.gradient_yy(gimg)


blobs = find_bright_keypoints(xx,xy,yy)


t1,t2 = partition_pixels(blobs,2)
peaks = find_peaks(blobs, non_max_suppression_radius=15, thresh=t2)

win4 = image_window(jet(blobs))

win = image_window(img)
win.add_overlay(centered_rects(peaks, 50, 50))

input("hit enter to stop")
