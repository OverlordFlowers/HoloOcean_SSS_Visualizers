import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import cv2
import random as rng

def objects_seen_by_waypoint(auv_pose, object_bboxes, sonar_range):
    auv_x, auv_y, auv_z, auv_yaw = map(float, auv_pose)  # Ensure float type
    detected_objects = []

    # Compute perpendicular sonar scan line direction (90-degree rotated from yaw)
    perpendicular_direction = np.array([
        -np.sin(np.radians(auv_yaw)),  # X component (perpendicular to yaw)
         np.cos(np.radians(auv_yaw))   # Y component (perpendicular to yaw)
    ], dtype=np.float64)

    # Compute start and end points of the sonar scan line
    left_endpoint = np.array([auv_x, auv_y]) - sonar_range * perpendicular_direction
    right_endpoint = np.array([auv_x, auv_y]) + sonar_range * perpendicular_direction

    for i, bbox in enumerate(object_bboxes):
        is_detected = False

        for point in bbox:
            px, py = map(float, point)  # Ensure float type

            # Compute if the point is within the perpendicular sonar scan range
            projection_length = np.dot(np.array([px, py]) - np.array([auv_x, auv_y]), perpendicular_direction)

            if -sonar_range <= projection_length <= sonar_range:
                is_detected = True
                break  # No need to check other points in the bounding box

        if is_detected:
            detected_objects.append(i)

    return detected_objects

def checkBoundingBoxCollision(directory):
    metadata = ((pd.read_csv(directory+"metadata.csv")).to_numpy())[0]

    scans = pd.read_csv(directory+"scan.csv").to_numpy()
    print(scans.shape)
    bounding_boxes_df = pd.read_csv(directory+"bounding_boxes.csv")
    bounding_boxes = []
    for _, row in bounding_boxes_df.iterrows():
        bbox = [(row[f'x{i}'], row[f'y{i}']) for i in range(4)]  # Extract bounding box points
        bounding_boxes.append(bbox)

    min_range   = metadata[0]
    max_range   = metadata[1]
    num_bins    = metadata[2]
    az          = metadata[3]
    map_size    = int(metadata[4])

    x = scans[:,1]
    print(len(x))
    for i in range(10):
        print(x[i])
    y = scans[:,2]
    z = scans[:,3]
    yaw = scans[:,6]
    
    
    for i in range(len(x)):
        detected = objects_seen_by_waypoint([x[i], y[i], z[i], yaw[i]], bounding_boxes, 10.0)
        print(detected)
    



checkBoundingBoxCollision("outputs/20250219154912/")


