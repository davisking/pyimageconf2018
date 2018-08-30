import sys
sys.path = ['./superfast/build'] + sys.path

import dlib
import numpy as np
import superfast


random_projections = np.random.randn(100,75*75).astype('float32');
random_projections = np.asmatrix(random_projections)

def hash_image(filename):
    img = dlib.load_grayscale_image(filename)
    img = dlib.convert_image(img, dtype='float32')
    img = dlib.resize_image(img, 75,75)
    img = img.reshape(img.size,1)
    img = np.asmatrix(img)
    img -= 110

    h = random_projections*img;
    h = h>0;
    return hash(np.packbits(h).tostring())

for filename in sys.argv[1:]:
    h = hash_image(filename)
    print("{} \t{}".format(h, filename))

# for h,f in superfast.hash_images(sys.argv[1:]):
   # print(h, "\t", f)

# for h,f in superfast.hash_images_parallel(sys.argv[1:]):
   # print(h, "\t", f)


# Time this program with a statement like:
#   time find images/small_face_dataset/ -name "*.png" | xargs python3 020_hash_images.py | wc


