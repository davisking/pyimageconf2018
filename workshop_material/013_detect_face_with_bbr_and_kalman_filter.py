
# What if the bounding box regression still isn't good enough?  It was still a
# little wobbly.  Well, since this is a video we can use a Kalman filter to
# smooth the box's position over time.  That's what we do in this example.


import dlib
import pickle




import cv2
cap = cv2.VideoCapture('images/moving_face.m4v');

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('bbr_predictor.dat')
win = dlib.image_window()

rect = dlib.rectangle()

def shape_to_rect(shape):
    r = dlib.rectangle()
    for p in shape.parts():
        r += p 
    return r

def learn_rect_filter():
    track = dlib.rectangles();
    while True:
        retval, frame = cap.read()
        if not retval:
            print("hit end of video file!")
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        dets = detector(img)
        for d in dets:
            shape = predictor(img, d)
            track.append(shape_to_rect(shape))

        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)

    rf = dlib.find_optimal_rect_filter(track,smoothness=10);
    print(rf)

    with open('rect_filter.pickle', 'wb') as handle:
        pickle.dump(rf, handle)


#learn_rect_filter()
#cap.release()
#exit(0)


rf = pickle.load(open('rect_filter.pickle','rb'))
# The above load just does this for the provided rect_filter.pickle file.
#rf = dlib.rect_filter(measurement_noise=32.6884, typical_acceleration=0.0420467, max_measurement_deviation=0.15212)
print(rf)
# 
#  dlib.rect_filter uses Kalman filters internally.  To explain this let's
#  suppose there is an object that moves according to this model:
#     position_{i+1} = position_i + velocity_i
#     velocity_{i+1} = velocity_i + some_unpredictable_acceleration
#
#  Moreover, assume all you get to see are position measurements that are
#  corrupted by Gaussian noise.  If some_unpredictable_acceleration is also
#  Gaussian distributed then you can find the optimal estimates of position_i
#  and velocity_i using a Kalman filter.  That's what dlib.rect_filter does
#  internally and the parameters above tell the Kalman filter how much
#  measurement noise and unpredictable acceleration are present.  There is also
#  a meta-rule in the rect_filter that prevents the filtered position from ever
#  being more than max_measurement_deviation*measurement_noise pixels away from
#  the last measured position.   
#
#  For further reading see "An introduction to the Kalman Filter" by Greg Welch and Gary Bishop 




while True:
    retval, frame = cap.read()
    if not retval:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    dets = detector(img)

    dets2 = dlib.rectangles();
    for d in dets:
        shape = predictor(img, d)
        #dets2.append(shape_to_rect(shape))
        dets2.append(rf(shape_to_rect(shape)))


    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets2)


cap.release()









# What if it's too slow?
#  0. Realize the detector might be upsampling the image, is detector(img,0) faster?
#  1. Downsample the image by half using dlib.pyramid_down(2).
#  2. Crop out a little window around the previous detection and only run the detector on that crop.
#     So replace the detector call with something like this:
    # if prev_det:
        # crop_rect = dlib.grow_rect(prev_det,20).intersect(dlib.get_rect(img))
        # dets = detector(dlib.sub_image(img, crop_rect))
        # dets = [dlib.translate_rect(d,crop_rect.tl_corner()) for d in dets]
    # else:
        # dets = detector(img)
    # prev_det = dets[0]
#  3. Don't run detector every frame, instead use the bounding box regression model to follow the face.
    # if prev_det:
        # shape = predictor(img, prev_det)
        # dets = [shape_to_rect(shape)]
    # else:
        # dets = detector(img)
        # prev_det = dets[0]


#  4. Train a nuclear norm version of the detector, see the next example programs for that.

