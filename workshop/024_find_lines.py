
from dlib import *
import numpy as np


img = load_grayscale_image('images/find_page/paper40.jpg')

pyr = pyramid_down(2)

img = convert_image(img, dtype='float32')
img = pyr(pyr(img))

# make the image super noisy
# img += 150*np.random.randn(*img.shape)

# img = gaussian_blur(img, 2);

win = image_window(img)

use_sobel = True 
if use_sobel:
    horz,vert = sobel_edge_detector(img)
else:
    ig = image_gradients(8)
    xx = ig.gradient_xx(img)
    xy = ig.gradient_xy(img)
    yy = ig.gradient_yy(img)
    horz,vert = find_dark_lines(xx,xy,yy)
    
lines = suppress_non_maximum_edges(horz,vert)


win2 = image_window(jet(lines))


win3 = image_window(hysteresis_threshold(lines))

# t1, t2, t3 = partition_pixels(lines,3)
# win3 = image_window(hysteresis_threshold(lines, t2, t3))

input("hit enter to stop")

