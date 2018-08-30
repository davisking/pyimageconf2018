

# Another thing you can do to make a detector that runs faster is to enable
# nuclear norm regularization.  What's that?  This will cause the learned HOG
# filters to be separable, or almost separable.  This example trains just such
# a HOG detector.



import dlib

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4
options.be_verbose = True

# Nuclear norm!!!!!!!!!!!!!!!!!!
options.nuclear_norm_regularization_strength = 1


training_data_filename = "images/dlib_example_faces/training.xml" 

dlib.train_simple_object_detector(training_data_filename, "detector_nuclear.svm", options)



print("Training accuracy: ", dlib.test_simple_object_detector(training_data_filename, "detector_nuclear.svm"))


detector = dlib.simple_object_detector("detector_nuclear.svm")

print("num separable filters: ", dlib.num_separable_filters(detector))
print("num separable filters thresh 0.1: ", dlib.num_separable_filters(dlib.threshold_filter_singular_values(detector,0.1)))
print("num separable filters thresh 1.0: ", dlib.num_separable_filters(dlib.threshold_filter_singular_values(detector,1.0)))

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)
win_det.wait_until_closed()

