import sys
sys.path = ['./superfast/build'] + sys.path

from dlib import *
del range
import numpy as np
import cv2
import superfast


win = image_window()
win_mbd = image_window()
win_skel = image_window()

def raster_scan(img, dist, lower, upper):
    area = shrink_rect(get_rect(img),1);

    def check_neighbor(r, c, neighbor_r, neighbor_c):
        nonlocal lower
        nonlocal upper 
        nonlocal dist 
        l = min(lower[neighbor_r][neighbor_c], img[r][c]);
        u = max(upper[neighbor_r][neighbor_c], img[r][c]);
        d = u-l;
        if (d < dist[r][c]):
            lower[r][c] = l
            upper[r][c] = u
            dist[r][c] = d

    # scan top to bottom
    for r in range(area.top(), area.bottom()+1):
        for c in range(area.left(), area.right()+1):
            check_neighbor(r,c, r-1,c)
            check_neighbor(r,c, r,c-1)

    # scan top to bottom
    for r in reversed(range(area.top(), area.bottom()+1)):
        for c in reversed(range(area.left(), area.right()+1)):
            check_neighbor(r,c, r+1,c)
            check_neighbor(r,c, r,c+1)

    # scan left to right 
    for c in range(area.left(), area.right()+1):
        for r in range(area.top(), area.bottom()+1):
            check_neighbor(r,c, r-1,c)
            check_neighbor(r,c, r,c-1)

    # scan right to left 
    for c in reversed(range(area.left(), area.right()+1)):
        for r in reversed(range(area.top(), area.bottom()+1)):
            check_neighbor(r,c, r+1,c)
            check_neighbor(r,c, r,c+1)

def mbd(img):
    img = img.astype('float32')
    # make dist the same size as img and filled with inf values, except for the borders, which are 0.
    dist = np.full(img.shape, float('inf'), dtype='float32')
    zero_border_pixels(dist,1,1)

    lower = img.copy()
    upper = img.copy()

    for i in range(3):
        # raster_scan(img, dist, lower, upper)
        superfast.raster_scan(img, dist, lower, upper)

    return dist


cam = cv2.VideoCapture('images/find_spinner/spinner.m4v')
while True:
    retval, img = cam.read()
    if not retval:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    img, valid_area = gaussian_blur(as_grayscale(img), 1.0)

    img = sub_image(img, valid_area)

    #dist = min_barrier_distance(img)
    dist = mbd(img)
    win_mbd.set_image(jet(dist))
    skel = skeleton(threshold_image(dist))
    win_skel.set_image(skel)
    labels, num_blobs = label_connected_blobs(skel)

    # group line endpoints together when they are part of the same blob
    objs = [[] for _ in range(num_blobs)]
    for p in find_line_endpoints(skel):
        objs[labels[p.y][p.x]].append(p)

    win.clear_overlay()
    win.set_image(img)
    for obj in objs:
        if len(obj) == 3:
            win.add_overlay(line(obj[0],obj[1]))
            win.add_overlay(line(obj[1],obj[2]))
            win.add_overlay(line(obj[2],obj[0]))

            radius = round(length(obj[0]-obj[1])/3.0)
            win.add_overlay_circle(obj[0], radius)
            win.add_overlay_circle(obj[1], radius)
            win.add_overlay_circle(obj[2], radius)

    input("hit enter to continue")



