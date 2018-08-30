

from dlib import *
from math import sqrt



img = load_grayscale_image('images/find_galaxies/nasa_crop.jpg')

win = image_window(img)

# instead of thresholding we can use a watershed, which is much more appropriate in this case
img = gaussian_blur(img, 0.5);
labels,num_blobs = label_connected_blobs_watershed(img,
                                                   background_thresh=partition_pixels(img),
                                                   smoothing=2);

print("num_blobs: {}".format(num_blobs))

win2 = image_window(randomly_color_image(labels))



rects = rectangles(num_blobs)
for r in range(labels.shape[0]):
    for c in range(labels.shape[1]):
        if labels[r][c] != 0:
            rects[labels[r][c]] += point(c,r)

win.add_overlay(rects)

# You can also crop out the galaxies, presumably to pass them to some classifier.
# Depending on what you are doing it might be appropriate to use boxes that preserve aspect
# ratios in the cropped images, such as this:
sqrects = [centered_rect(r,int(1.5*sqrt(r.area())),int(1.5*sqrt(r.area()))) for r in rects]
    
# make crop plans that will all be 40x40 pixels in size
dets = [chip_details(r,chip_dims(40,40)) for r in rects]
#dets = [chip_details(r,40*40) for r in rects]

win3 = image_window(tile_images(extract_image_chips(img, dets)))


input("hit enter to continue")

