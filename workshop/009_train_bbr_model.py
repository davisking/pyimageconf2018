
# Now let's train the bounding box regression model using the dataset we just made.

import dlib


training_xml_file = "images/small_face_dataset/faces_600_bbr.xml" 

options = dlib.shape_predictor_training_options()
options.num_threads = 4
options.be_verbose =  True 

# I'll explain how I selected these magic numbers in a few minutes.  
options.cascade_depth = 7
options.tree_depth = 2
options.num_trees_per_cascade_level = 277
options.nu = 0.0326222
options.oversampling_amount = 20
options.oversampling_translation_jitter = 0.181914
options.feature_pool_size = 400
options.lambda_param = 0.14798
options.num_test_splits = 20
options.feature_pool_region_padding = 0.108275



dlib.train_shape_predictor(training_xml_file, "bbr_predictor.dat", options)
print("\nTraining error: ", dlib.test_shape_predictor(training_xml_file, "bbr_predictor.dat"))


