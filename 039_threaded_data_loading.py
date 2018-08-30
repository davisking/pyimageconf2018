import sys
sys.path = ['./superfast/build'] + sys.path

import superfast
import dlib
import timeit
import random



files = sys.argv[1:]

start = timeit.default_timer()
for i in range(1000):
    img = dlib.load_rgb_image(random.choice(files))
stop = timeit.default_timer()
print("time to load images: ", stop-start)



start = timeit.default_timer()
data_loader = superfast.threaded_data_loader(files, 4)
for i in range(1000):
    img = data_loader.get_next_image()
stop = timeit.default_timer()
print("time to load images superfast: ", stop-start)
