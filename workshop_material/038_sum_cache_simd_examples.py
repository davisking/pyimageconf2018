import numpy as np
import sys
sys.path = ['./superfast/build'] + sys.path
import superfast


# To see what this code does you should open ipython and just run these
# commands and time time using %time.  Also go into the superfast code and
# uncomment the loop that makes these routines do many internal iterations.
# Observe the times.


img = np.random.randn(256,1024*4)

img = np.random.randn(256,1024*4).astype('float32')

superfast.sum_row_major_order(img)

superfast.sum_row_major_order_simd(img)

# this is like 3x slower, but if you change the number of columns even a little it's not so bad. why?
superfast.sum_column_major_order(img)

