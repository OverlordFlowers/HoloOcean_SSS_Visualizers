# merge occupancy maps together
# get bounding boxes for each object
# get all waypoints that see the objects
# optimize using TSP
# should be easy, right?
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

from helper_functions.getthething import *
from helper_functions.populate_bbregions import *

resolution = 0.1
map_size = 100


def findDirectoriesWithMarker(start_dir, marker):
    """
    Walk through all subdirectories starting from 'start_dir'.
    If a subdirectory's name contains the marker, print its path.
    """
    directories = []
    for root, dirs, files in os.walk(start_dir):
        for d in dirs:
            if marker in d:
                directories.append(d)
    return directories

#marker = "m"
#directories = find_directories_with_marker("./outputs/", marker)

def compileOccupancy(directories, map_size, resolution, show_plot=False):
    occ_grid = None
    n = len(directories)
    n_half = int(np.ceil(n/2))
    if (show_plot is True):
        plt.figure()
        f, axarr = plt.subplots(n_half, n_half)
        plt.gca().invert_yaxis()
    for i in range(len(directories)):
        df = pd.read_csv("outputs/"+directories[i]+"/occupancy_grid_pp.csv")
        arr = df.to_numpy()[:,1:]
        
        if (show_plot is True):
            axarr[i//n_half][i%n_half].set_title(f"m{i}", fontsize=28)
            axarr[i//n_half][i%n_half].tick_params(axis='both', labelsize=24)
            axarr[i//n_half][i%n_half].imshow(arr, origin='lower', cmap='copper', extent=[-map_size/2, map_size/2, -map_size/2, map_size/2])

        if (occ_grid is None):
            occ_grid = arr
        else:
            occ_grid = np.logical_or(occ_grid, arr)
    
    if (show_plot is True):
        plt.savefig("separate.pdf", format='pdf')
        plt.figure(figsize=(8, 8))
        plt.gca().invert_yaxis()
        plt.title("Composite Occupancy Map", fontsize=28)
        plt.xlabel("x (meters)", fontsize=24)
        plt.ylabel("y (meters)", fontsize=24)
        plt.tick_params(axis='both', labelsize=20)
        plt.imshow(occ_grid, origin='lower', cmap='copper', extent=[-map_size/2, map_size/2, -map_size/2, map_size/2])
        plt.savefig("composite.pdf", format='pdf')
        plt.show()

    return np.flipud(occ_grid)

def getBoundingBoxRegions(occupancy_grid, map_size, resolution, show_drawing=False):
    kernel = np.ones((5, 5), np.uint8)

    # Perform dilation
    # dilate then erod to clear noise
    dilation = cv2.dilate(occupancy_grid.astype(np.uint8), kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    erosion = erosion.astype(np.uint8)
    #erosion = cv2.flip(erosion, 0)

    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height,width = occupancy_grid.shape
    empty_image = np.zeros((height, width, 3), dtype=np.uint8)
    countoured_image = cv2.drawContours(empty_image, contours, -1, (0, 255, 0), 2)

    minRect = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)

    drawing = np.zeros((countoured_image.shape[0], countoured_image.shape[1], 3), dtype=np.uint8)

    directory = "outputs_composite/"
    # I already have the infrastructure in place for this and I'm feeling lazy.
    f = open(directory+"bounding_boxes.csv", "w")
    f.write("obj,x0,y0,x1,y1,x2,y2,x3,y3")

    boxes = []
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))

        # contour
        cv2.drawContours(drawing, contours, i, color)
        
        # rotated rectangle
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        boxes.append(box)
        cv2.drawContours(drawing, [box], 0, color)
        #cv2.putText(drawing, str(i), (box[0,0], box[0,1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        #box = (box * resolution) - (map_size/2.0)
        f.write("\n")
        f.write(f"{i},{box[0,0]},{box[0,1]},{box[1,0]},{box[1,1]},{box[2,0]},{box[2,1]},{box[3,0]},{box[3,1]}")
    f.close()

    boxes = (np.multiply(boxes, resolution)) - (map_size / 2.0)

    if (show_drawing):
        cv2.imshow('Contours', drawing)
        cv2.imwrite('Contours.png', drawing)
        cv2.waitKey(0)
    
    
    return boxes

def rasterizeBoundingBoxRegions(bounding_boxes, occupancy_grid, resolution, map_size, show_plot = False):
    map_shape = occupancy_grid.shape
    rasterized_grid = np.zeros((map_shape[0], map_shape[1]), dtype=np.uint8)

    bounding_boxes_scaled = ((bounding_boxes + (map_size/2))/resolution)

    for bbox in bounding_boxes_scaled:
        # Convert to grid coordinates
        grid_bbox = [(x, y) for x, y in bbox]
        #print(grid_bbox)
        # Fill polygon in the occupancy grid using OpenCV
        cv2.fillPoly(rasterized_grid, [np.array(grid_bbox, np.int32)], 1)

    if (show_plot):
        plt.figure(figsize=(8, 8))
        plt.title("Rasterized Bounding Box Map", fontsize=16)
        plt.xlabel("x (meters)", fontsize=12)
        plt.ylabel("y (meters)", fontsize=12)
        plt.tick_params(axis='both', labelsize=10)
        plt.imshow(rasterized_grid, cmap='copper',extent=[-map_size/2, map_size/2, -map_size/2, map_size/2])
        plt.savefig("rasterizedbb.svg", format='svg')
        plt.savefig("rasterizedbb")
        plt.show()

    return rasterized_grid

def getScans(directories):
    all_scans = []
    
    for i in range(len(directories)):
        d = "outputs/"+directories[i]+"/"
        scans = pd.read_csv(d+"scan.csv").to_numpy()

        auv_pos = scans[:, [1,2,3,6]]
        all_scans.append(auv_pos)

    return all_scans




def line_intersection(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def objects_seen_by_waypoint(auv_pose, object_bboxes, sonar_range):
    auv_x, auv_y, auv_z, auv_yaw = map(float, auv_pose)  # Ensure float type
    detected_objects = []

    perpendicular_direction = np.array([
        np.cos(np.radians(auv_yaw)),   # X component
        -np.sin(np.radians(auv_yaw))   # Y component
    ], dtype=np.float64)
    #print(perpendicular_direction)

    # Compute start and end points of the sonar scan line
    left_endpoint = np.array([auv_x, auv_y]) + sonar_range * perpendicular_direction
    right_endpoint = np.array([auv_x, auv_y]) - sonar_range * perpendicular_direction
    sonar_scan_line = (tuple(left_endpoint), tuple(right_endpoint))

    for i, bbox in enumerate(object_bboxes):
        is_detected = False

        bbox_edges = [(bbox[j], bbox[(j + 1) % 4]) for j in range(4)]

        for edge in bbox_edges:
            if line_intersection(sonar_scan_line[0], sonar_scan_line[1], edge[0], edge[1]):
                is_detected = True
                break

        if is_detected:
            detected_objects.append(i)
    if (detected_objects == []):
        detected_objects = None

    return detected_objects

def getScansDetected(scans, sonar_range, bounding_boxes, get_all=False):
    
    waypoints = []
    detected_obj = []

    for i in range(len(scans)):
        curr_wps = []
        curr_detected = []

        for j in range(len(scans[i])):
            detected = objects_seen_by_waypoint(scans[i][j], bounding_boxes, sonar_range)

            if (get_all and detected is None):
                detected = []

            if (detected is not None):
                curr_wps.append(scans[i][j])
                curr_detected.append(detected)

        waypoints.append(curr_wps)
        detected_obj.append(curr_detected)
        
        pass

    return waypoints, detected_obj


def compute_sidescan_coverage(waypoints, occupancy_grid, sonar_width):
    """
    Determines which pixels each waypoint covers using a side-scan sonar swath.

    Parameters:
    - waypoints: List of (x, y) coordinates.
    - occupancy_grid: 2D numpy array representing the rasterized map.
    - sonar_width: Maximum sonar scan width (perpendicular to AUV heading).
    - yaw_angles: List of yaw angles (degrees) for each waypoint.

    Returns:
    - coverage: Dictionary {waypoint_index: set(covered pixels)}.
    """
    coverage = {i: set() for i in range(len(waypoints))}
    grid_height, grid_width = occupancy_grid.shape

    for i, (wx, wy, wz, yaw) in enumerate(waypoints):
        gx, gy = int(wx), int(wy)  # Convert to grid indices
        
        # Compute perpendicular swath direction
        dx = np.cos(np.radians(yaw))
        dy = np.sin(np.radians(yaw))

        # Compute swath start and end points
        left_x, left_y = gx - sonar_width * dy, gy + sonar_width * dx
        right_x, right_y = gx + sonar_width * dy, gy - sonar_width * dx

        # Rasterize the line swath onto the grid using OpenCV
        scan_line = np.array([[int(left_x), int(left_y)], [int(right_x), int(right_y)]], np.int32)
        cv2.line(occupancy_grid, scan_line[0], scan_line[1], 1, thickness=1)

        # Store covered pixels
        for x in range(grid_width):
            for y in range(grid_height):
                if occupancy_grid[y, x] == 1:
                    coverage[i].add((x, y))

    return coverage

def merge_redundant_waypoints(waypoints, coverage):
    """
    Removes redundant waypoints that provide overlapping or minimal additional coverage.

    Parameters:
    - waypoints: List of (x, y) coordinates.
    - coverage: Dictionary {waypoint_index: set(covered pixels)}.

    Returns:
    - Filtered waypoints and updated coverage.
    """
    unique_waypoints = []
    unique_coverage = {}
    seen_coverages = set()

    for i, wp in enumerate(waypoints):
        print(len(coverage))
        coverage_tuple = tuple(sorted(coverage[i]))  # Convert to immutable form
        if coverage_tuple not in seen_coverages:
            seen_coverages.add(coverage_tuple)
            unique_waypoints.append(wp)
            unique_coverage[len(unique_waypoints) - 1] = coverage[i]

    return unique_waypoints, unique_coverage

def tsp_heuristic(waypoints, coverage):
    """
    Solves a modified TSP using a nearest-neighbor heuristic while maximizing coverage area.

    Parameters:
    - waypoints: List of (x, y) coordinates.
    - coverage: Dictionary {waypoint_index: set(covered pixels)}.

    Returns:
    - Optimized waypoint sequence.
    """
    n = len(waypoints)
    dist_matrix = cdist(waypoints, waypoints)
    unvisited = set(range(n))
    path = [0]  # Start at the first waypoint
    unvisited.remove(0)
    seen_pixels = set(coverage[0])  # Track scanned pixels

    while unvisited:
        last = path[-1]
        best_next = None
        best_score = -np.inf

        for candidate in unvisited:
            travel_cost = dist_matrix[last, candidate]
            new_pixels = coverage[candidate] - seen_pixels
            redundancy_penalty = len(coverage[candidate] & seen_pixels) * 0.5

            score = len(new_pixels) - redundancy_penalty - 0.1 * travel_cost

            if score > best_score:
                best_score = score
                best_next = candidate

        if best_next is None:
            break

        path.append(best_next)
        seen_pixels.update(coverage[best_next])
        unvisited.remove(best_next)

    return [waypoints[i] for i in path]


def rescale_waypoints(waypoints, map_size, resolution):
    """
    Rescales waypoints from real-world coordinates to the occupancy grid scale.

    Parameters:
    - waypoints: List of (x, y, z, yaw) coordinates in meters.
    - resolution: The resolution of the grid (meters per pixel).

    Returns:
    - Scaled waypoints in grid coordinates.
    """
    return [(int((x + map_size//2) / resolution), int((y + map_size//2) / resolution), int(z), yaw) for x, y, z, yaw in waypoints]

def compute_sidescan_coverage(waypoints, occupancy_grid, sonar_range, resolution):
    """
    Determines which pixels each waypoint covers using a side-scan sonar swath.

    Parameters:
    - waypoints: List of (x, y, z, yaw) coordinates in meters.
    - occupancy_grid: 2D numpy array representing the rasterized map.
    - sonar_range: Maximum sonar scan width in meters (extends perpendicularly).
    - resolution: The resolution of the grid (meters per pixel).

    Returns:
    - coverage: 2D numpy array where 1 represents covered areas.
    """
    grid_height, grid_width = occupancy_grid.shape
    coverage_grid = np.zeros_like(occupancy_grid, dtype=np.uint8)

    for (wx, wy, wz, yaw) in waypoints:
        gx, gy = wx, wy  # Convert to grid indices
        sonar_width = int(sonar_range / resolution)  # Convert sonar width to grid units

        # Compute perpendicular swath direction based on yaw
        dx = np.cos(np.radians(yaw))
        dy = np.sin(np.radians(yaw))

        # Compute swath start and end points (left and right of AUV)
        left_x, left_y = gx - sonar_width * dy, gy + sonar_width * dx
        right_x, right_y = gx + sonar_width * dy, gy - sonar_width * dx

        # Ensure points are within grid bounds
        left_x, left_y = max(0, min(grid_width - 1, int(left_x))), max(0, min(grid_height - 1, int(left_y)))
        right_x, right_y = max(0, min(grid_width - 1, int(right_x))), max(0, min(grid_height - 1, int(right_y)))

        # Rasterize the swath as a line onto the grid using OpenCV
        scan_line = np.array([[left_x, left_y], [right_x, right_y]], np.int32)
        cv2.line(coverage_grid, scan_line[0], scan_line[1], 1, thickness=1)

    return coverage_grid