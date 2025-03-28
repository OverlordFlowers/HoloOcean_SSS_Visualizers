import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
import cv2
import random as rng
from scipy.spatial.distance import cdist
from helper_functions.mission_composition import *
# look for all directories with specific markers pertaining to a mission
# (they should all use the same metadata, so we can load one and save it)
marker = "m"
output_directory = "outputs_composite/" + marker + "/"

marked_directories = findDirectoriesWithMarker("./outputs/", marker)

metadata = pd.read_csv("outputs/"+marked_directories[0]+"/metadata.csv")
min_range = int(metadata['min_range'].iloc[0]) # Something something deprecation
max_range = int(metadata['max_range'].iloc[0])
num_bins = int(metadata['num_bins'].iloc[0])
azimuth = int(metadata['azimuth'].iloc[0]) # Unused for now, could prove useful in the future
map_size = int(metadata['mapsize'].iloc[0])
resolution = float(metadata['resolution'].iloc[0])

# compile all the scans together into one
# generate a total occupancy map
occ_grid = compileOccupancy(marked_directories, map_size, resolution, show_plot=False)

# get the bounding boxes from the total occupancy map
bounding_boxes = getBoundingBoxRegions(occ_grid, map_size, resolution, show_drawing=False)

# get new occupancy map using the bounding boxes (rasterization), rescale to grid points instead of float coordinate values
rasterized_regions = rasterizeBoundingBoxRegions(bounding_boxes, occ_grid, resolution, map_size, show_plot=False)

# get all scans from the missions
scans = getScans(marked_directories)

waypoints, objects_seen = getScansDetected(scans, max_range, bounding_boxes, get_all=False)
all_waypoints = [wp for m in waypoints for wp in m]
#print(all_waypoints)
scaled_waypoints = rescale_waypoints(all_waypoints, map_size, resolution)

coverage_grid = compute_sidescan_coverage(scaled_waypoints, rasterized_regions, max_range, resolution)



start_time = time.time()
def optimize_waypoints_smooth(waypoints, coverage_grid, alpha=50, beta=0.0):
    n = len(waypoints)
    dist_matrix = cdist([(wp[0], wp[1]) for wp in waypoints], [(wp[0], wp[1]) for wp in waypoints])
    unvisited = set(range(n))
    path = [0] 
    unvisited.remove(0)
    current_yaw = waypoints[0][3]  
    
    while unvisited:
        last = path[-1]
        best_next = None
        best_score = -np.inf

        for candidate in unvisited:
            travel_cost = dist_matrix[last, candidate]
            x, y, z, yaw = waypoints[candidate]
            
            new_coverage = np.sum(coverage_grid[max(0, y - 1):min(y + 2, coverage_grid.shape[0]),
                                                max(0, x - 1):min(x + 2, coverage_grid.shape[1])])

            yaw_change_penalty = beta * abs(yaw - current_yaw)

            score = new_coverage - alpha * travel_cost - yaw_change_penalty

            if score > best_score:
                best_score = score
                best_next = candidate

        if best_next is None:
            break

        path.append(best_next)
        current_yaw = waypoints[best_next][3]  # Update heading
        unvisited.remove(best_next)

    return [waypoints[i] for i in path]

def interpolate_waypoints(waypoints, max_distance):
    interpolated_waypoints = [waypoints[0]]  # Start with the first waypoint

    for i in range(len(waypoints) - 1):
        x1, y1, z1, yaw1 = waypoints[i]
        x2, y2, z2, yaw2 = waypoints[i + 1]

        # Compute Euclidean distance
        dist = np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])
        print(dist)

        if dist > max_distance:
            num_points = int(np.ceil(dist / max_distance))

            x_interp = np.linspace(x1, x2, num_points + 1)
            y_interp = np.linspace(y1, y2, num_points + 1)
            z_interp = np.linspace(z1, z2, num_points + 1)
            yaw_interp = np.linspace(yaw1, yaw2, num_points + 1)

            for j in range(1, num_points):
                x_prev, y_prev = x_interp[j - 1], y_interp[j - 1]
                x_next, y_next = x_interp[j + 1] if j + 1 < num_points else x2, y_interp[j + 1] if j + 1 < num_points else y2

                yaw_new = np.degrees(np.arctan2(y_next - y_prev, x_next - x_prev))
                print(num_points)
                interpolated_waypoints.append([x_interp[j], y_interp[j], z_interp[j], yaw_new])

        interpolated_waypoints.append([x2, y2, z2, yaw2])  
    
    adjusted_waypoints = interpolated_waypoints.copy()

    # Not pretty but I needed to make a second pass to make sure the yaws were correct
    for i in range(len(adjusted_waypoints) - 1):
        x1, y1, z1, _ = adjusted_waypoints[i]
        x2, y2, z2, _ = adjusted_waypoints[i + 1]
        y1 = -y1
        y2 = -y2

        delta_x = x2 - x1
        delta_y = y2 - y1

        if delta_x == 0 and delta_y == 0:
            yaw_corrected = adjusted_waypoints[i - 1][3] if i > 0 else 0  
        elif delta_x == 0:  
            yaw_corrected = 90.0 if delta_y > 0 else -90.0
        elif delta_y == 0:  
            yaw_corrected = 0.0 if delta_x > 0 else 180.0  
        else:
            yaw_corrected = np.degrees(np.arctan2(delta_y, delta_x))

        # Update the yaw in the waypoint list
        print(yaw_corrected)
        adjusted_waypoints[i] = (x1, -y1, z1, yaw_corrected)
    

    return adjusted_waypoints

# **Run the optimized smooth waypoint planner**
#smoothed_waypoints = scaled_waypoints
optimized_smooth_waypoints = optimize_waypoints_smooth(scaled_waypoints, coverage_grid)
end_time = time.time()

max_interpolation_distance = 1
smoothed_waypoints = interpolate_waypoints(optimized_smooth_waypoints, max_interpolation_distance)


f = open("optimized_waypoint.csv", "w")
f.write("x,y,z,yaw")

for p in smoothed_waypoints:
    f.write("\n")
    f.write(f"{p[0]*resolution - map_size/2},{p[1]*resolution-map_size/2},{p[2]},{p[3]}")

print(f"Time to solve: {end_time - start_time}")

#print(optimized_smooth_waypoints)

import matplotlib.cm as cm


optimized_x = [int(wp[0]) for wp in smoothed_waypoints]
optimized_y = [int(wp[1]) for wp in smoothed_waypoints]

plot_interval = 100
filtered_x = optimized_x[::plot_interval]
filtered_y = optimized_y[::plot_interval]

coverage_grid = compute_sidescan_coverage(smoothed_waypoints, rasterized_regions, max_range, resolution)

plt.figure(figsize=(8, 8))
plt.imshow(occ_grid, cmap="gray", origin="upper", alpha=0.5, extent=[-map_size/2, map_size/2, -map_size/2, map_size/2])
plt.imshow(coverage_grid, cmap="Blues", origin="upper", alpha=0.5, extent=[-map_size/2, map_size/2, -map_size/2, map_size/2])  # Overlay sonar coverage

num_waypoints = len(smoothed_waypoints)
colors = cm.viridis(np.linspace(0, 1, num_waypoints))
# Plot waypoints with color gradient based on order
for i, (x, y) in enumerate(zip(filtered_x, filtered_y)):
    plt.scatter(x*resolution - map_size//2, -y*resolution+map_size//2, color=colors[i], marker="o", label=f"WP {i+1}" if i in [0, num_waypoints-1] else "_nolegend_")

for i in range(num_waypoints - 1):
    plt.plot([optimized_x[i]*resolution - map_size//2, optimized_x[i+1]*resolution - map_size//2], [-optimized_y[i]*resolution + map_size//2, -optimized_y[i+1]*resolution + map_size//2], color=colors[i], linestyle="--")

plt.title("Optimized Mission Occupancy Map", fontsize=28)
plt.xlabel("x (meters)", fontsize=24)
plt.ylabel("y (meters)", fontsize=24)
plt.tick_params(axis='both', labelsize=16)
#plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
#plt.colorbar(cm.ScalarMappable(cmap="viridis"), label="Waypoint Order")
plt.savefig("optimized.svg", format='svg')
plt.show()