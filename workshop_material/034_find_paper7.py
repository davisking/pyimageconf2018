from dlib import *
import numpy as np

import sys
sys.path = ['./superfast/build'] + sys.path
import superfast
import tools
import timeit


# This version now works on all the test images!

# img = load_grayscale_image(sys.argv[1])

img = load_grayscale_image('./images/find_page/tissue_01.jpg')
# img = load_grayscale_image('./images/find_page/paper21.jpg')

ht = hough_transform(300)

img = sub_image(gaussian_blur(img,3)) 
img = resize_image(img, ht.size, ht.size)

win1 = image_window(img)

img = equalize_histogram(img) 
x,y = sobel_edge_detector(img)
edges = suppress_non_maximum_edges(x,y)
win3 = image_window(edges)
normalize_image_gradients(x,y)                    
superfast.discard_wacky_edge_groups(edges, x, y)  
edges = hysteresis_threshold(edges, partition_pixels(edges), 50)  
win4 = image_window(edges)


# !!!!! NEW !!!! Do a coherent Hough transform rather than the regular Hough transform.
ht2 = tools.hough_transform(ht.size)
start = timeit.default_timer()
himg = tools.coherent_hough_transform(ht2, edges, x, y)
# himg = superfast.coherent_hough_transform(ht, edges, x, y)
#himg = ht(edges)
stop = timeit.default_timer()
print("coherent hough transform time: ", stop-start)


hits = ht.find_strong_hough_points(himg, hough_count_thresh=ht.size/5, angle_nms_thresh=15, radius_nms_thresh=10)

# Look at the lines the Hough transform found.  Observe that it finds way fewer
# when using the coherent version.
lines = [ht.get_line(l) for l in hits]
win3.add_overlay(lines)


print("hough hits found: ", len(hits))
# keep just the 50 best lines
if (len(hits) > 50):
    hits = hits[0:50]

# NEW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Use tools.find_hough_boxes(), which still checks if all the corners are
# connected, but does so in a more nuanced way.
line_pixels = ht.find_pixels_voting_for_lines(edges, hits, 4,4)
# NEW, remove obviously bogus line pixels by checking if their angle makes sense.
for i in range(len(line_pixels)):
    line_pixels[i] = remove_incoherent_edge_pixels(line_pixels[i], x, y, angle_thresh=40)
boxes = tools.find_hough_boxes(ht, hits, line_pixels)


if len(boxes) > 0:
    c1,c2,c3,c4,area,idx1,idx2,idx3,idx4 = boxes[0]
    win1.add_overlay(line(c1,c2))
    win1.add_overlay(line(c2,c3))
    win1.add_overlay(line(c3,c4))
    win1.add_overlay(line(c4,c1))

    page = extract_image_4points(img, [c1,c2,c3,c4], 200,200)
    win_page = image_window(page)

    cimg = convert_image(edges, dtype='rgb_pixel')
    for l in [line_pixels[i] for i in [idx1, idx2, idx3, idx4]]:
        for p in l:
            cimg[p.y,p.x,:] = (255,0,0)
    win4.set_image(cimg)

input("hit enter to exit")

