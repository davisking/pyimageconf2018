

from dlib import *




img = load_grayscale_image('images/find_galaxies/nasa_crop.jpg')

win = image_window(img)

binary_image = threshold_image(img)

win2 = image_window(binary_image)

labels,num_blobs = label_connected_blobs(binary_image);

print("num_blobs: {}".format(num_blobs))

win3 = image_window(randomly_color_image(labels))


# Put a rectangle around each blob.
rects = rectangles(num_blobs)
for r in range(labels.shape[0]):
    for c in range(labels.shape[1]):
        if labels[r][c] != 0:
            rects[labels[r][c]] += point(c,r)

win.add_overlay(rects)


input("hit enter to continue")

