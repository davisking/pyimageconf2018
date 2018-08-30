import sys
sys.path = ['./superfast/build'] + sys.path
from dlib import *
from math import sqrt
import superfast
import timeit

img = load_grayscale_image('images/find_galaxies/nasa_crop.jpg')

win = image_window(img)

# instead of thresholding we can use a watershed, which is much more appropriate in this case
img = gaussian_blur(img, 0.5);
labels,num_blobs = label_connected_blobs_watershed(img,
                                                   background_thresh=partition_pixels(img),
                                                   smoothing=2);

print("num_blobs: {}".format(num_blobs))

win2 = image_window(randomly_color_image(labels))

start_time = timeit.default_timer()
# This loop is super slow, doing it in C++ is way faster
rects = rectangles(num_blobs)
for r in range(labels.shape[0]):
    for c in range(labels.shape[1]):
        if labels[r][c] != 0:
            rects[labels[r][c]] += point(c,r)
# rects = superfast.blobs_to_rects(labels, num_blobs)
total_time = timeit.default_timer()-start_time

print("time: {}".format(total_time))

win.add_overlay(rects)

# could also use these rectangles since they preserve aspect ratios
sqrects = [centered_rect(r,int(1.5*sqrt(r.area())),int(1.5*sqrt(r.area()))) for r in rects]
    
dets = [chip_details(r,chip_dims(40,40)) for r in rects]
#dets = [chip_details(r,40*40) for r in rects]

win3 = image_window(tile_images(extract_image_chips(img, dets)))


input("hit enter to continue")

