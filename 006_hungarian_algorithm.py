from dlib import *
import numpy as np
import matplotlib.pyplot as plt



v1 = [point(38, 28),
 point(62, 114),
 point(81, 191),
 point(113, 269),
 point(142, 344),
 point(219, 266),
 point(265, 188),
 point(280, 103),
 point(317, 37)]

v2 = [point(201, 32),
 point(240, 196),
 point(217, 110),
 point(261, 266),
 point(356, 259),
 point(289, 332),
 point(429, 116),
 point(390, 193),
 point(463, 56)]


dists = matrix(len(v1),len(v2))
for i in range(len(v1)):
    for j in range(len(v2)):
        d = length(v1[i] - v2[j])
        dists[i][j] = -d*d

assignments = max_cost_assignment(dists)

print(assignments)



plt.figure()
for i in range(len(assignments)):
    j = assignments[i]
    # Uncomment to see the assignments
    #plt.plot([v1[i].x, v2[j].x], [v1[i].y, v2[j].y], 'ro-', zorder=1)
    plt.scatter(v1[i].x, v1[i].y, c='g', zorder=2)
    plt.scatter(v2[j].x, v2[j].y, c='b', zorder=2)


plt.show()
