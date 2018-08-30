

import sys
import dlib

# Train on the images in dlib's examples/faces folder.  But first, let's make
# our own dataset by using imglab to annotate the images ourselves. 
training_data_filename = sys.argv[1]



options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4
options.be_verbose = True
# what do the outputs of verbose training mean?  In particular, what is the risk and risk gap?



dlib.train_simple_object_detector(training_data_filename, "detector.svm", options)



print("Training accuracy: ", dlib.test_simple_object_detector(training_data_filename, "detector.svm"))


detector = dlib.simple_object_detector("detector.svm")

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)
win_det.wait_until_closed()

