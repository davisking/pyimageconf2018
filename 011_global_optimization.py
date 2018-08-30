
# So what about those magic parameters?  How did I determine them?  
# Well, I used a hyperparameter optimizer!
#
# This example introduces the optimizer I used and we will talk about how it works in some detail.  


import dlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from math import sin,cos,pi,exp,sqrt




# Let's fine the maximizer of this horrible function:
def messy_holder_table(x,y):
    Z = abs(sin(x)*cos(y)*exp(abs(1-sqrt(x*x+y*y)/pi)))
    R = max(9, abs(x)+abs(y))**5
    return 1e5*Z / R

xy,z = dlib.find_max_global(messy_holder_table, 
                           [-15,-15],  # Lower bound constraints on x and y respectively
                           [15,15],    # Upper bound constraints on x and y respectively
                           100)        # The number of times find_min_global() will call messy_holder_table()

print("xy: ", xy);
print("z: ", z);
opt_z = messy_holder_table(-8.162150706931659, 0)
print("distance from optimal: ", opt_z - z)








# Now plot a 3D view of messy_holder_table() and also draw the point the optimizer located
X = np.arange(-15, 15, 0.1)
Y = np.arange(-15, 15, 0.1)
X, Y = np.meshgrid(X, Y)

from numpy import sin,cos,pi,exp,sqrt
Z = abs(sin(X)*cos(Y)*exp(abs(1-sqrt(X*X+Y*Y)/pi)))
R = np.maximum(9,abs(X) + abs(Y))**5
Z = 1e5*Z/R

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=70, ccount=70,
                       linewidth=0, antialiased=True)


# Put a green dot on the location found by dlib.find_max_global()
ax.scatter(xy[0],xy[1], z, s=40, c='g')

plt.show()
