
# One problem you might have is the detection boxes not being placed accurately
# enough.  For example, the HOG detector has an 8 pixel stride, which is
# reasonable for many problems, but not all.  This example runs the detector on
# a video so we can clearly see the effect of the stride.

import dlib
import cv2

cap = cv2.VideoCapture('images/moving_face.m4v');

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

while True:
    retval, frame = cap.read()
    if not retval:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    dets = detector(img)

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)

    input("hit enter to continue")

cap.release()

