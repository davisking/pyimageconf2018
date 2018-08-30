
import dlib
import sys

detector = dlib.simple_object_detector("detector.svm")
sp = dlib.shape_predictor("landmark_predictor.dat")


win = dlib.image_window()

for f in sys.argv[1:]:
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

    dets = detector(img,1)
    for d in dets:
        shape = sp(img,d)
        win.add_overlay(shape)

    input("Hit enter to continue")



# go run imglab --flip to double your training dataset and try it again!
