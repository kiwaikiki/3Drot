import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib import colors as mcolors
import colorsys

def man_cmap(cmap, value=1.):
    colors = cmap(np.arange(cmap.N))
    hls = np.array([colorsys.rgb_to_hls(*c) for c in colors[:,:3]])
    hls[:,1] *= value
    rgb = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    return mcolors.LinearSegmentedColormap.from_list("", rgb)

cmap = plt.cm.get_cmap("cool")

cmap = man_cmap(cmap, 1.1)


# Generate random points
points = np.random.rand(1000, 2)

# points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)

# Create Voronoi diagram
vor = Voronoi(points)

# Plot Voronoi diagram
fig, ax = plt.subplots(figsize=(8, 8))

# Plot Voronoi regions
for region_index in range(len(vor.regions)):
    region = vor.regions[region_index]
    if not -1 in region and region:
        polygon = [vor.vertices[i] for i in region]
        color = plt.cm.prism(np.random.rand(1))  # Use centroid position for color assignment
        # plt.fill(*zip(*polygon), color=color)
        plt.fill(*zip(*polygon), color=cmap(region_index / len(vor.regions)))
    

# # # Plot Voronoi vertices
# plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')  

# # # Plot input points
# plt.plot(points[:, 0], points[:, 1], 'o')

plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
