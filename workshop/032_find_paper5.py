from dlib import *
import numpy as np

import sys
sys.path = ['./superfast/build'] + sys.path
import superfast
import tools




#img = load_grayscale_image(sys.argv[1])

img = load_grayscale_image('./images/find_page/paper32.jpg')

# What about this image?  Need to do something to fix it
# img = load_grayscale_image('./images/find_page/paper01.jpg')

ht = hough_transform(300)

img = sub_image(gaussian_blur(img,3)) 
img = resize_image(img, ht.size, ht.size)

win1 = image_window(img)

img = equalize_histogram(img) # NEW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
x,y = sobel_edge_detector(img)
edges = suppress_non_maximum_edges(x,y)
win3 = image_window(edges)
normalize_image_gradients(x,y)                    
superfast.discard_wacky_edge_groups(edges, x, y)  
edges = hysteresis_threshold(edges, partition_pixels(edges), 50)  # NEW, lowered thresholds so we get way more edges 
win4 = image_window(edges)
himg = ht(edges)


hits = ht.find_strong_hough_points(himg, hough_count_thresh=ht.size/5, angle_nms_thresh=15, radius_nms_thresh=10)

# NEW, now we are using tools.find_hough_boxes_simple() to find all the plausible boxes and
# selecting the largest one.
print("hough hits found: ", len(hits))
# keep just the 50 best lines
if (len(hits) > 50):
    hits = hits[0:50]
boxes = tools.find_hough_boxes_simple(ht, hits)


if len(boxes) > 0:
    c1,c2,c3,c4,area,idx1,idx2,idx3,idx4 = boxes[0]
    win1.add_overlay(line(c1,c2))
    win1.add_overlay(line(c2,c3))
    win1.add_overlay(line(c3,c4))
    win1.add_overlay(line(c4,c1))

    page = extract_image_4points(img, [c1,c2,c3,c4], 200,200)
    win_page = image_window(page)

input("hit enter to exit")

