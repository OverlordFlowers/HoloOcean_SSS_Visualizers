import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import cv2
import random as rng

# I don't think we really need to differentiate between which regions belong to which objects
# only that it covers it
def rasterize_bounding_boxes(bounding_boxes, map_size, resolution):

    grid_width = int(map_size / resolution)
    print(grid_width)
    occupancy_grid = np.zeros((grid_width, grid_width), dtype=np.uint8)

    for bbox in bounding_boxes:
        # Convert to grid coordinates
        grid_bbox = [(int(x + map_size/2)/resolution, int(y + map_size/2)/resolution) for x, y in bbox]
        print(grid_bbox)
        # Fill polygon in the occupancy grid using OpenCV
        cv2.fillPoly(occupancy_grid, [np.array(grid_bbox, np.int32)], 1)

    return occupancy_grid


def get_boundingbox_coverage(directory):
    metadata = pd.read_csv(directory+"metadata.csv")
    min_range = int(metadata['min_range'].iloc[0]) # Something something deprecation
    max_range = int(metadata['max_range'].iloc[0])
    num_bins = int(metadata['num_bins'].iloc[0])
    azimuth = int(metadata['azimuth'].iloc[0]) # Unused for now, could prove useful in the future
    map_size = int(metadata['mapsize'].iloc[0])
    resolution = float(metadata['resolution'].iloc[0])

    df = pd.read_csv(directory+"bounding_boxes.csv")
    bounding_boxes_df = pd.read_csv(directory+"bounding_boxes.csv")
    bounding_boxes = []
    for _, row in bounding_boxes_df.iterrows():
        bbox = [(row[f'x{i}'], row[f'y{i}']) for i in range(4)] 
        bounding_boxes.append(bbox)

    occ_grid = rasterize_bounding_boxes(bounding_boxes, map_size, resolution)
    print(occ_grid)

    # Step 3: Visualize the grid
    plt.figure(figsize=(6, 6))
    plt.imshow(occ_grid, cmap="gray", origin="lower", extent=[-map_size/2, map_size/2, -map_size/2, map_size/2])
    plt.title("Rasterized Bounding Boxes (Occupancy Grid)", fontsize=16)
    plt.xlabel("x (meters)", fontsize=12)
    plt.ylabel("y (meters)", fontsize=12)
    plt.tick_params(axis='both', labelsize=10)
    plt.show()

    return occ_grid
    pass

#directory="outputs/20250219154912/"

#get_boundingbox_coverage(directory)
