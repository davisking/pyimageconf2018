

# Multiple HOG detectors are better than one.  In this example we will see the
# limits of a single HOG detector and overcome it by training a group of 5
# detectors.  


import dlib

options = dlib.simple_object_detector_training_options()


options.add_left_right_image_flips = True 
options.C = 5
options.num_threads = 4
options.be_verbose = True
dlib.train_simple_object_detector("images/small_face_dataset/faces_5poses.xml", "detector_all.svm", options)

# It's under-fitting on the training data.  So the model might not be powerful
# enough, or C not large enough.  What happens when C gets bigger? In this case
# it fits the training data but creates a lot of false alarms on testing data.
print("Training_all accuracy: ", dlib.test_simple_object_detector("images/small_face_dataset/faces_5poses.xml", 
                                                                  "detector_all.svm"))



print("go make a multi-HOG training dataset using imglab --cluster 5 !!")
exit(0)

# our dataset is oriented, so definitely don't add in flips.
options.add_left_right_image_flips = False 
options.C = 10 
options.num_threads = 4
options.be_verbose = True
options.nuclear_norm_regularization_strength = 1


dlib.train_simple_object_detector("images/small_face_dataset/cluster_001.xml", "detector1.svm", options)
dlib.train_simple_object_detector("images/small_face_dataset/cluster_002.xml", "detector2.svm", options)
dlib.train_simple_object_detector("images/small_face_dataset/cluster_003.xml", "detector3.svm", options)
dlib.train_simple_object_detector("images/small_face_dataset/cluster_004.xml", "detector4.svm", options)
dlib.train_simple_object_detector("images/small_face_dataset/cluster_005.xml", "detector5.svm", options)


print("Training1 accuracy: ", dlib.test_simple_object_detector("images/small_face_dataset/cluster_001.xml", "detector1.svm"))
print("Training2 accuracy: ", dlib.test_simple_object_detector("images/small_face_dataset/cluster_002.xml", "detector2.svm"))
print("Training3 accuracy: ", dlib.test_simple_object_detector("images/small_face_dataset/cluster_003.xml", "detector3.svm"))
print("Training4 accuracy: ", dlib.test_simple_object_detector("images/small_face_dataset/cluster_004.xml", "detector4.svm"))
print("Training5 accuracy: ", dlib.test_simple_object_detector("images/small_face_dataset/cluster_005.xml", "detector5.svm"))



