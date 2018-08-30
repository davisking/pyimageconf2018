
# Try running this on the files images/dlib_example_faces/*.jpg You should
# notice the multi-HOG detector (the red boxes) works a little better than the
# single model HOG detector.

import sys
import dlib


detector1 = dlib.simple_object_detector("detector1.svm")
detector2 = dlib.simple_object_detector("detector2.svm")
detector3 = dlib.simple_object_detector("detector3.svm")
detector4 = dlib.simple_object_detector("detector4.svm")
detector5 = dlib.simple_object_detector("detector5.svm")

detectors = [detector1, detector2, detector3, detector4, detector5]

for d in detectors:
    print("num_separable_filters = ", dlib.num_separable_filters(dlib.threshold_filter_singular_values(d,0.01)))

detectors = [dlib.threshold_filter_singular_values(d,0.01) for d in detectors]

# if all the detectors use identical sliding window sizes (i.e. if
# detector.detection_window_height and detector.detection_window_width are the
# same for all the detectors) then you can pack them into one detector object.
detector = dlib.simple_object_detector(detectors)
print("upsampling = " , detector.upsampling_amount)
detector.upsampling_amount = 1

#detector.save("combined_det.svm")
#detector = dlib.simple_object_detector('combined_det.svm')


detector_all = dlib.simple_object_detector('detector_all.svm')
#detector_all = dlib.simple_object_detector('detector.svm')
print("upsampling = " , detector.upsampling_amount)

win = dlib.image_window()

for f in sys.argv[1:]:
    print(f)
    img = dlib.load_rgb_image(f)

    dets = detector(img,2)
    dets_all = detector_all(img,2)
    #dets, scores, idxs = dlib.simple_object_detector.run_multiple(detectors, img, 2)

    # grow these rectangles slightly so they don't overlap dets in the image_window
    dets_all = [dlib.grow_rect(d,1) for d in dets_all]

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets_all, dlib.rgb_pixel(0,255,0))
    win.add_overlay(dets)

    input("hit enter to continue")

