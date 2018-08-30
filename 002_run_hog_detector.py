

import sys
import dlib


detector = dlib.simple_object_detector("detector.svm")

win = dlib.image_window()

for f in sys.argv[1:]:
    img = dlib.load_rgb_image(f)

    dets = detector(img)

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)

    input("hit enter to continue")

