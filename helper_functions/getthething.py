import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import cv2
import random as rng

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

def checkBoundingBoxCollision(directory, write_all=False):
    metadata = ((pd.read_csv(directory+"metadata.csv")).to_numpy())[0]

    scans = pd.read_csv(directory+"scan.csv").to_numpy()
    bounding_boxes_df = pd.read_csv(directory+"bounding_boxes.csv")
    bounding_boxes = []
    for _, row in bounding_boxes_df.iterrows():
        bbox = [(row[f'x{i}'], row[f'y{i}']) for i in range(4)] 
        bounding_boxes.append(bbox)

    min_range   = metadata[0]
    max_range   = metadata[1]
    num_bins    = metadata[2]
    az          = metadata[3]
    map_size    = int(metadata[4])

    x = scans[:,1]
    y = scans[:,2]
    z = scans[:,3]
    yaw = scans[:,6]
    
    f = open(directory+"wp_detected_objects.csv", "w")
    f.write("x,y,z,yaw,objects_detected")
    for i in range(len(x)):
        
        detected = objects_seen_by_waypoint([x[i], y[i], z[i], yaw[i]], bounding_boxes, max_range)

        if (write_all or detected is not None):
            f.write("\n")
            f.write(f"{x[i]},{y[i]},{z[i]},{yaw[i]},{str(detected).replace(',', ' ')}")

    f.close()

#checkBoundingBoxCollision("outputs/20250219154912/")


