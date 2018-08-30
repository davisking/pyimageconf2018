

import dlib
import cv2
import timeit, sys

cap = cv2.VideoCapture('images/moving_face.m4v');

#detector = dlib.simple_object_detector("detector.svm")
detector = dlib.simple_object_detector("detector_nuclear.svm")

print("num separable filters = ", dlib.num_separable_filters(detector))
detector = dlib.threshold_filter_singular_values(detector,1)
print("num separable filters after threshold = ", dlib.num_separable_filters(detector))
win = dlib.image_window()


start = 0
stop = 0

while True:
    retval, frame = cap.read()
    if not retval:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start = timeit.default_timer()
    dets = detector(img)
    stop = timeit.default_timer()
    print("time = ", stop-start)
    sys.stdout.flush()

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)

    #input("hit enter to continue")

cap.release()


# The real benefit of this kind of thing is when you run many detectors though.  E.g. how fast is this?
many_detectors = [detector for i in range(10)]
dets, scores, idxs = dlib.simple_object_detector.run_multiple(many_detectors, img, upsample_num_times=1)

