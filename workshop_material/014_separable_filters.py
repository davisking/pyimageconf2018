
# You need to know what a separable filter is to understand this "nuclear norm" thing.
# So what is a separable filter?  Let's find out.

import numpy as np
import dlib
from math import sqrt


# make a separable filter
a = np.matrix([1, 5, 8, 5, 1])
filt = a.transpose()*a

print("Filter:\n", filt)

# You can use the SVD to decompose the filter back into its separable parts
[u,w,v] = np.linalg.svd(filt)

print("\nSVD outputs:")
print(u)
print(v)
print("\nsingular values: ", w)

# Use the SVD output to get the filter again.  Note that the print shows the same thing as print(filt)
b = v[0] * sqrt(w[0])
print("\nFilter built from SVD outputs:")
print(b.transpose()*b)



# To drive this home a little further, lets run a few tests.  We are going to
# filter this image a few different ways.  All of these different ways of
# filtering the image give the same outputs. The only difference if if we take
# advantage of the separability of the filter or not.
img = dlib.load_grayscale_image('images/testing_faces/2007_001430.jpg')

# Do all the filtering with float values
filt = filt.astype('float32')
a = a.astype('float32')
img = img.astype('float32')


# These all output the same things, except the last version is faster. 
fimg1,valid_area = dlib.spatially_filter_image(img, filt)
fimg2,valid_area = dlib.spatially_filter_image(dlib.spatially_filter_image(img, a), a.transpose().copy())
fimg3,valid_area = dlib.spatially_filter_image_separable(img, a, a)


print("")
print("filter output difference: ", np.max(np.abs(fimg1 - fimg2)))
print("filter output difference: ", np.max(np.abs(fimg1 - fimg3)))


