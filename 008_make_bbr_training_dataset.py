

# One easy way to improve box placement accuracy is to train a shape_predictor
# model to tell you the exact position of the box.  So in this example we make
# a training dataset appropriate for creating just such a shape_predictor
# model.

import dlib
import os


data = dlib.image_dataset_metadata.load_image_dataset_metadata('images/small_face_dataset/faces_600.xml')

print(data)

detector = dlib.get_frontal_face_detector()

os.chdir('images/small_face_dataset')
all_dets = []
for img in data.images:
    dets = detector(dlib.load_rgb_image(img.filename),1)
    all_dets.append(dets)
    print("found {} faces in {}".format(len(dets), img.filename))

data = dlib.make_bounding_box_regression_training_data(data, all_dets)
dlib.image_dataset_metadata.save_image_dataset_metadata(data, 'faces_600_bbr.xml')

