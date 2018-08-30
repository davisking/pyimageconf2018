from dlib import *
import numpy as np

import sys
sys.path = ['./superfast/build'] + sys.path
import superfast
import tools




#img = load_grayscale_image(sys.argv[1])

img = load_grayscale_image('./images/find_page/paper30.jpg')

# What about this image?  Need to do something to fix it
# img = load_grayscale_image('./images/find_page/paper32.jpg')

ht = hough_transform(300)

img = sub_image(gaussian_blur(img,3)) 
img = resize_image(img, ht.size, ht.size)

win1 = image_window(img)

x,y = sobel_edge_detector(img)
edges = suppress_non_maximum_edges(x,y)
win3 = image_window(edges)
normalize_image_gradients(x,y)                    # NEW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
tools.discard_wacky_edge_groups(edges, x, y)  # NEW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
edges = hysteresis_threshold(edges)  # NEW, no more discard_all_but_largest_blob()
win4 = image_window(edges)
himg = ht(edges)

hits = ht.find_strong_hough_points(himg, hough_count_thresh=ht.size/5, angle_nms_thresh=15, radius_nms_thresh=10)
lines = [ht.get_line(p) for p in hits[0:4]]

win1.add_overlay(lines)
page = extract_image_4points(img, lines, 200,200)
win_page = image_window(page)

input("hit enter to exit")

