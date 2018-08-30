

# Now let's run the detector with the bounding box regression model and see if
# the detections look smoother.

import dlib
import cv2

cap = cv2.VideoCapture('images/moving_face.m4v');

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('bbr_predictor.dat')
win = dlib.image_window()

def shape_to_rect(shape):
    r = dlib.rectangle()
    for p in shape.parts():
        r += p 
    return r

while True:
    retval, frame = cap.read()
    if not retval:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    dets = detector(img)
    dets2 = [];
    for d in dets:
        shape = predictor(img, d)
        dets2.append(shape_to_rect(shape))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets2)

    #input("hit enter to continue")

cap.release()

