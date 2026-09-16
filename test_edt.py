import numpy as np
from scipy.ndimage import distance_transform_edt

grid = np.zeros((10, 10))
grid[5, 5] = 100
dists = distance_transform_edt(grid < 50)
print(dists)
