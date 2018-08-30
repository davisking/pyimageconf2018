
import numpy as np
from math import pi,cos,sin,sqrt
from dlib import point, get_rect, center
import dlib

###########################################################################################

class hough_transform:
    def __init__(self, size):
        self.size = size

    def perform_generic_hough_transform(self, img, record_hit):
        assert(img.shape[0] == self.size)
        assert(img.shape[1] == self.size)


        cent = center(get_rect(img))
        even_size = self.size - (self.size%2)
        sqrt_2 = sqrt(2)

        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                val = img[r][c]
                if (val != 0):
                    x = c - cent.x
                    y = r - cent.y
                    # Now draw the curve in Hough space for this image point
                    for t in range(self.size):
                        theta = t*pi/even_size
                        radius = (x*cos(theta) + y*sin(theta))/sqrt_2 + even_size/2 + 0.5
                        rr = int(radius)
                        record_hit(point(t,rr), point(c,r), val)


    def __call__(self, img):
        himg = np.zeros(img.shape, dtype='float32')

        def record_hit(hough_point, img_point, value):
            nonlocal himg
            himg[hough_point.y][hough_point.x] += value

        self.perform_generic_hough_transform(img, record_hit)

        return himg


###########################################################################################

def coherent_hough_transform(ht, edges, horz, vert):
    hcoherent = np.zeros((ht.size, ht.size, 3), dtype='float32')

    def record_hit(hough_point, img_point, value):
        x = horz[img_point.y][img_point.x]
        y = vert[img_point.y][img_point.x]

        # accumulate hessian matrices 
        hcoherent[hough_point.y][hough_point.x][0] += x*x 
        hcoherent[hough_point.y][hough_point.x][1] += x*y 
        hcoherent[hough_point.y][hough_point.x][2] += y*y 

    ht.perform_generic_hough_transform(edges, record_hit)

    himg = np.zeros((ht.size, ht.size), dtype='float32')
    for r in range(himg.shape[0]):
        for c in range(himg.shape[1]):
            ev = real_eigenvalues(hcoherent[r][c][0], hcoherent[r][c][1], hcoherent[r][c][2])
            if (max(ev) != 0 and min(ev)/max(ev) < 0.30):
                himg[r][c] = max(ev)
            else:
                himg[r][c] = 0

    return himg


###########################################################################################

def label_blobs_with_similar_angles(img, horz, vert, angle_threshold):
    labels = np.zeros(img.shape, dtype='uint32')

    dotprod_angle_thresh = cos(angle_threshold*pi/180)

    next_label = 1
    area = get_rect(img)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            # skip already labeled pixels or background pixels
            if (labels[r][c] != 0 or img[r][c] == 0):
                continue

            labels[r][c] = next_label

            # now label all the connected neighbors of this point
            neighbors = [(c,r)]
            while len(neighbors) > 0:
                x,y = neighbors.pop()

                window = [(x-1,y-1), (x,y-1), (x+1,y-1),
                         (x-1,y),            (x+1,y),
                         (x-1,y+1), (x,y+1), (x+1,y+1)]
                for xx,yy in window:
                    # If this neighbor is in the image, not background, and not already labeled
                    if (area.contains(xx,yy) and img[yy][xx]!=0 and labels[yy][xx]==0):
                        dotprod = horz[y][x]*horz[yy][xx] + vert[y][x]*vert[yy][xx]
                        # if the angle between these two vectors is less than angle_threshold degrees.
                        if dotprod > dotprod_angle_thresh:
                            labels[yy][xx] = next_label
                            neighbors.append((xx,yy))

            next_label += 1

    return labels, next_label


###########################################################################################

def discard_wacky_edge_groups (edges, horz, vert):
    labels, num_blobs = label_blobs_with_similar_angles(edges, horz, vert, 25)
    blob_sizes = dlib.get_histogram(labels, num_blobs)
    # blank out short edges
    for r in range(edges.shape[0]):
        for c in range(edges.shape[1]):
            if blob_sizes[labels[r][c]] < 20:
                edges[r][c] = 0

###########################################################################################

def real_eigenvalues(xx, xy, yy):
    "Return the eigenvalues of the matrix [xx xy; xy yy]"
    b = -(xx + yy)
    c = xx*yy - xy*xy

    disc = b*b - 4*c
    if (disc >= 0):
        disc = sqrt(disc)
    else:
        disc = 0

    v0 = (-b + disc)/2
    v1 = (-b - disc)/2
    return (v0,v1)


###########################################################################################

from dlib import intersect, angle_between_lines, polygon_area, count_points_on_side_of_line
from dlib import count_points_between_lines, length

def find_hough_boxes_simple(ht, hits):

    # convert hough coordinates into lines in original image
    lines = [ht.get_line(h) for h in hits]

    angle_thresh = 20 # in degrees

    def are_parallel(a,b):
        intersects_outside_image = not get_rect(ht).contains(intersect(a,b))
        return angle_between_lines(a,b) < angle_thresh and intersects_outside_image

    # find all the parallel lines
    parallel = []
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            if are_parallel(lines[i], lines[j]):
                parallel.append((lines[i], lines[j], i, j))

    def line_separation(a,b):
        center1 = (a.p1+a.p2)/2
        center2 = (b.p1+b.p2)/2
        return length(center1-center2)

    # sort the parallel line pairs so that lines that are most separated come first:
    parallel = sorted(parallel, key=lambda a : line_separation(a[0],a[1]), reverse=True)

    print("number of parallel line pairs: ", len(parallel))


    boxes = []
    area = get_rect(ht)
    # Now find boxes, these are pairs of parallel lines where all the intersecting points
    # are contained within the original image.

    for i in range(len(parallel)):
        for j in range(i+1,len(parallel)):

            l1,l3, idx1,idx3 = parallel[i]
            l2,l4, idx2,idx4 = parallel[j]

            c1 = intersect(l1,l2)
            c2 = intersect(l2,l3)
            c3 = intersect(l3,l4)
            c4 = intersect(l4,l1)

            # skip this pair if it's outside the image
            if (not area.contains(c1) or
                not area.contains(c2) or
                not area.contains(c3) or
                not area.contains(c4) ):
                continue

            polyarea = polygon_area([c1, c2, c3, c4])

            boxes.append((c1,c2,c3,c4,polyarea,idx1,idx2,idx3,idx4))

    boxes = sorted(boxes, key=lambda x : x[4], reverse=True)
    return boxes
        
###########################################################################################

def find_hough_boxes_less_simple(ht, hits, line_pixels):
    assert(len(hits) == len(line_pixels))


    boxes = []
    for box in find_hough_boxes_simple(ht, hits):

        c1,c2,c3,c4,polyarea,idx1,idx2,idx3,idx4 = box

        pix1 = line_pixels[idx1]
        pix2 = line_pixels[idx2]
        pix3 = line_pixels[idx3]
        pix4 = line_pixels[idx4]

        l1 = ht.get_line(hits[idx1])
        l2 = ht.get_line(hits[idx2])
        l3 = ht.get_line(hits[idx3])
        l4 = ht.get_line(hits[idx4])


        center = (c1 + c2 + c3 + c4)/4

        # check if all the corners are connected to each other
        dist = 20
        num_required = 15
        if (count_points_on_side_of_line(l1, center, pix2, 1, dist) >= num_required and
            count_points_on_side_of_line(l2, center, pix1, 1, dist) >= num_required and
            count_points_on_side_of_line(l3, center, pix4, 1, dist) >= num_required and
            count_points_on_side_of_line(l4, center, pix3, 1, dist) >= num_required and
            count_points_on_side_of_line(l2, center, pix3, 1, dist) >= num_required and
            count_points_on_side_of_line(l3, center, pix2, 1, dist) >= num_required and
            count_points_on_side_of_line(l4, center, pix1, 1, dist) >= num_required and
            count_points_on_side_of_line(l1, center, pix4, 1, dist) >= num_required):
            boxes.append((c1,c2,c3,c4,polyarea,idx1,idx2,idx3,idx4))

    return boxes
        
###########################################################################################


def find_hough_boxes(ht, hits, line_pixels):
    assert(len(hits) == len(line_pixels))


    boxes = []
    for box in find_hough_boxes_simple(ht, hits):

        c1,c2,c3,c4,polyarea,idx1,idx2,idx3,idx4 = box

        pix1 = line_pixels[idx1]
        pix2 = line_pixels[idx2]
        pix3 = line_pixels[idx3]
        pix4 = line_pixels[idx4]

        l1 = ht.get_line(hits[idx1])
        l2 = ht.get_line(hits[idx2])
        l3 = ht.get_line(hits[idx3])
        l4 = ht.get_line(hits[idx4])


        center = (c1 + c2 + c3 + c4)/4

        def corners_connected(l1,l2, pix1,pix2):

            if len(pix1) == 0 or len(pix2) == 0:
                return False

            dist = 20
            corners_touch = False

            pts_in_l2_next_to_corner = count_points_on_side_of_line(l1, center, pix2, 1, dist)
            pts_in_l1_next_to_corner = count_points_on_side_of_line(l2, center, pix1, 1, dist)

            l2_near_corner = pts_in_l2_next_to_corner >= 15 
            l1_near_corner = pts_in_l1_next_to_corner >= 15 
            corners_touch = l1_near_corner and l2_near_corner

            corner = intersect(l1,l2)
            point_outside_box = 2*(corner-center) + center

            l2_near_corner = pts_in_l2_next_to_corner >= 5 
            l1_near_corner = pts_in_l1_next_to_corner >= 5 
            # The two lines are connected if they touch or if none of them
            # extends outside the bounds of the rectangle and at least one of
            # them goes up to the edge of the rectangle.
            return corners_touch or (count_points_on_side_of_line(l1, point_outside_box, pix2,2)/len(pix2) < 0.03 and
                                     count_points_on_side_of_line(l2, point_outside_box, pix1,2)/len(pix1) < 0.03 and
                                     (l1_near_corner or l2_near_corner))
        

        if (corners_connected(l1,l2,pix1,pix2) and
            corners_connected(l2,l3,pix2,pix3) and
            corners_connected(l3,l4,pix3,pix4) and
            corners_connected(l4,l1,pix4,pix1)):
            boxes.append((c1,c2,c3,c4,polyarea,idx1,idx2,idx3,idx4))

    return boxes
        
###########################################################################################

