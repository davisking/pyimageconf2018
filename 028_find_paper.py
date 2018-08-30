from dlib import *


import sys

#img = load_grayscale_image(sys.argv[1])

# Works well on these images
img = load_grayscale_image('./images/find_page/paper24.jpg')
# img = load_grayscale_image('./images/find_page/paper01.jpg')
# img = load_grayscale_image('./images/find_page/paper02.jpg')
# img = load_grayscale_image('./images/find_page/paper03.jpg')
# img = load_grayscale_image('./images/find_page/paper20.jpg')
# img = load_grayscale_image('./images/find_page/paper21.jpg')
# img = load_grayscale_image('./images/find_page/paper40.jpg')
# img = load_grayscale_image('./images/find_page/paper41.jpg')
# img = load_grayscale_image('./images/find_page/tissue_01.jpg')
# img = load_grayscale_image('./images/find_page/tissue_02.jpg')
# img = load_grayscale_image('./images/find_page/tissue_03.jpg')

# What about this image?  Need to do something to fix it
# img = load_grayscale_image('./images/find_page/paper22.jpg')

# Or what about these images?  They are even harder!
# img = load_grayscale_image('./images/find_page/tissue_04.jpg')
# img = load_grayscale_image('./images/find_page/paper32.jpg')


ht = hough_transform(300)
img = resize_image(img, ht.size, ht.size)

win1 = image_window(img)

ig = image_gradients(10)
x = ig.gradient_x(img)
y = ig.gradient_y(img)
edges = suppress_non_maximum_edges(x,y)
win3 = image_window(edges)
edges = hysteresis_threshold(edges)
win4 = image_window(edges)
himg = ht(edges)

hits = ht.find_strong_hough_points(himg, hough_count_thresh=ht.size/5, angle_nms_thresh=15, radius_nms_thresh=10)
lines = [ht.get_line(p) for p in hits[0:4]]

win1.add_overlay(lines)
page = extract_image_4points(img, lines, 200,200)
win_page = image_window(page)

input("hit enter to exit")

