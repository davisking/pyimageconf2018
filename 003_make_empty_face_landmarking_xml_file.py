

import sys
import dlib

# Now let's run our detector on the images in dlib's examples/faces folder and
# save the detections to XML. Then we can use imglab to annotate face landmarks
# for each of those detections.

detector = dlib.simple_object_detector("detector.svm")


dataset = dlib.image_dataset_metadata.dataset()

dataset.name = 'pyimageconf face landmarking dataset'

for f in sys.argv[1:]:
    img = dlib.load_rgb_image(f)
    image_metadata = dlib.image_dataset_metadata.image()
    image_metadata.filename = f

    dets = detector(img,1)
    for det in dets:
        box = dlib.image_dataset_metadata.box()
        box.rect = det
        image_metadata.boxes.append(box)

    dataset.images.append(image_metadata)

dlib.image_dataset_metadata.save_image_dataset_metadata(dataset, 'face_landmarking.xml')

