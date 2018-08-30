from dlib import *
import numpy as np

import sys
sys.path = ['./superfast/build'] + sys.path
import superfast


def discard_all_but_largest_blob(img):
    labels, num_blobs = label_connected_blobs(img, connected_if_both_not_zero=True)
    h = get_histogram(labels, num_blobs)
    # ignore background blobs
    h[0] = 0
    largest_blob = np.argmax(h)
    superfast.zero_pixels_not_labeled_with_val(img, labels, largest_blob)
    return img


#img = load_grayscale_image(sys.argv[1])

img = load_grayscale_image('./images/find_page/tissue_04.jpg')

# What about this image?  Need to do something to fix it
# img = load_grayscale_image('./images/find_page/paper30.jpg')

ht = hough_transform(300)

# resize_image is resizing the image so much it's introducing artifacts that
# mess up edge detection.  Blurring a little before resize fixes this.
img = sub_image(gaussian_blur(img,3)) # NEW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
img = resize_image(img, ht.size, ht.size)

win1 = image_window(img)

x,y = sobel_edge_detector(img) # NEW !!!!!!!!!!!!! (basically the same as using image_gradients(1))
edges = suppress_non_maximum_edges(x,y)
win3 = image_window(edges)
edges = discard_all_but_largest_blob(hysteresis_threshold(edges))  
win4 = image_window(edges)
himg = ht(edges)

hits = ht.find_strong_hough_points(himg, hough_count_thresh=ht.size/5, angle_nms_thresh=15, radius_nms_thresh=10)
lines = [ht.get_line(p) for p in hits[0:4]]

win1.add_overlay(lines)
page = extract_image_4points(img, lines, 200,200)
win_page = image_window(page)

input("hit enter to exit")

