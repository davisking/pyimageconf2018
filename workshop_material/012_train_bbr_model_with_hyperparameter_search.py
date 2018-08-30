


# So now let's use that hyperparameter optimizer to find the best parameters
# for our bounding box regression model.

import sys
import dlib




# Before we can run this example we have to make a train and test dataset.  So
# make these two files by using imglab --shuffle and --split-train-test
training_xml_path = "images/small_face_dataset/bbr_train.xml" 
testing_xml_path = "images/small_face_dataset/bbr_test.xml" 



def test_params(cascade_depth, padding, nu, tree_depth, num_trees_per_cascade_level, lambda_param, jitter):
    options = dlib.shape_predictor_training_options()
    options.feature_pool_region_padding = padding
    options.cascade_depth = int(cascade_depth)
    options.nu = nu
    options.tree_depth = int(tree_depth)
    options.oversampling_translation_jitter = jitter

    options.num_trees_per_cascade_level = int(num_trees_per_cascade_level)
    options.lambda_param = lambda_param 
    options.num_threads = 4
    options.be_verbose = True 


    print("start training")
    print(options)
    sys.stdout.flush()
    dlib.train_shape_predictor(training_xml_path, "bbr_predictor.dat", options)
    print("\nTraining error: ", dlib.test_shape_predictor(training_xml_path, "bbr_predictor.dat"))
    err = dlib.test_shape_predictor(testing_xml_path, "bbr_predictor.dat")
    print("\nTesting error: ", err)
    sys.stdout.flush()
    return err



lower = [5, -0.2, 0.001, 2, 100, 0.01, 0]
upper = [25, 0.2, 0.20,  5, 1000, 0.99, 0.3]
isint = [True, False, False, True, True, False, False]

x,y = dlib.find_min_global(test_params, 
                           bound1=lower,  
                           bound2=upper,    
                           is_integer_variable=isint,
                           num_function_calls=100)         

print("optimal inputs: {}".format(x));
print("optimal output: {}".format(y));

test_params(x[0],x[1],x[2],x[3],x[4],x[5],x[6])

