
# This example uses maximally stable extremal regions to find text.  We also
# compare it to MBD.

import cv2, dlib

# img = dlib.load_rgb_image('./images/toys.jpg')
img = dlib.load_rgb_image('./images/welcome_to_california.jpg')


gray = dlib.as_grayscale(img)

mser = cv2.MSER_create()
regions,boxes = mser.detectRegions(gray)

hulls = [cv2.convexHull(p) for p in regions]

# Draw the convex hulls onto img
dets = cv2.polylines(img.copy(), hulls, isClosed=True, color=(255,0,0),thickness=1);

win1 = dlib.image_window(img,  'original image')
win2 = dlib.image_window(dets, 'MSER detections')




mbd = dlib.min_barrier_distance(gray)
win3 = dlib.image_window(mbd, 'MBD')
win4 = dlib.image_window(dlib.threshold_image(dlib.gaussian_blur(mbd,2)), 'MBD thresholded')


input('hit enter to exit')


