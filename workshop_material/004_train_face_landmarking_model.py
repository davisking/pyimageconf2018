
import dlib


options = dlib.shape_predictor_training_options()
options.feature_pool_region_padding = 0.1 
options.cascade_depth = 10
options.landmark_relative_padding_mode = False
options.nu = 0.05
options.tree_depth = 2 
options.oversampling_translation_jitter = 0.1 
options.oversampling_amount = 200
options.num_threads = 4
options.be_verbose = True 


dlib.train_shape_predictor("face_landmarking.xml", "landmark_predictor.dat", options)
print("\nTraining error: ", dlib.test_shape_predictor("face_landmarking.xml", "landmark_predictor.dat"))


