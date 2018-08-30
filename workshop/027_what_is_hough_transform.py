

from dlib import *
import numpy as np
import tools

# img = load_grayscale_image('./images/hough_transform_test0.png')
img = load_grayscale_image('./images/hough_transform_test1.png')
# img = load_grayscale_image('./images/hough_transform_test2.png')

win = image_window(img)

# A C++ version 
ht = hough_transform(img.shape[0])
# A python version for reference (super slow, don't use for real)
ht_python = tools.hough_transform(img.shape[0])

# img += convert_image((np.random.rand(400,400)>0.5)*255, dtype='uint8')
# win_noisy = image_window(img)

himg = ht(img)

winh = image_window(himg)

l = ht.get_line(max_point(himg))

win.add_overlay(l)


input("hit enter to terminate")
