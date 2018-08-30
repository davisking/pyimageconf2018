
# This example takes about a minute to run and finds better training parameters
# than those we used in 017_train_multiple_hog_detectors.py.  Again, we do this
# using dlib.find_max_global().
#

import dlib
import sys




def test_params(C, nuclear_norm):
    options = dlib.simple_object_detector_training_options()
    # our dataset is oriented, so definitely don't add in flips.
    options.add_left_right_image_flips = False 
    options.C = C
    options.num_threads = 1
    options.be_verbose = False 
    options.nuclear_norm_regularization_strength = nuclear_norm
    options.max_runtime_seconds = 5 # SET REALLY SMALL SO THE DEMO DOESN'T TAKE TO LONG, USE BIGGER VALUES FOR REAL USE!!!!!

    dlib.train_simple_object_detector("images/small_face_dataset/cluster_001_train.xml", "detector1_.svm", options)

    # You can do a lot here.  Run the detector through
    # dlib.threshold_filter_singular_values() for instance to make sure it
    # learns something that will work once thresholded. We can also add a
    # penalty for having a lot of filters.   Run this program a few times and
    # try out different ways of penalizing the return from test_params() and
    # see what happens.
    result = dlib.test_simple_object_detector("images/small_face_dataset/cluster_001_test.xml", "detector1_.svm")
    print("C = {}, nuclear_norm = {}".format(C, nuclear_norm))
    print("testing accuracy: ",result)
    sys.stdout.flush()
    # For settings with the same average precision, we should prefer smaller C
    # since smaller C has better generalization.  
    return result.average_precision - C*1e-8



lower = [0.01, 0]
upper = [100, 10]

x,y = dlib.find_max_global(test_params, 
                           bound1=lower,  
                           bound2=upper,    
                           num_function_calls=20)         

print("optimal inputs: {}".format(x));
print("optimal output: {}".format(y));

test_params(x[0],x[1])



