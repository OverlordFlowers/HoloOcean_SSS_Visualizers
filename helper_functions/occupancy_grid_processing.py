import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
from skimage.filters import threshold_isodata
import cv2
import random as rng

def processOccupancyGrid(directory):
    # Replace with load metadata
    resolution = 0.1
    map_size = 100

    dataframe = pd.read_csv(directory+"occupancy_grid.csv")
    occupancy_grid = dataframe.to_numpy()
    occupancy_grid = occupancy_grid[:,1:]
    occupancy_grid[occupancy_grid<0.05] = 0.10

    # from occupancy grid
    # denoise image (from noisy sonar)
    blurred = gaussian_filter(occupancy_grid, sigma=0)

    thresh = (threshold_otsu(blurred)) 
    print(thresh)

    plt.imshow(blurred)
    plt.title("Blurred")
    plt.show()
    
    # threshold to isolate objects # *2 gets rid of noise from sonar
    
    binary = blurred > thresh
    binary = binary.astype('float64')
    plt.figure(dpi=1200)
    plt.title("Binary Occupancy Map")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.imshow(binary, cmap='gray', origin='lower', extent=[-map_size/2, map_size/2, -map_size/2, map_size/2])
    plt.savefig(directory+"binarymap")
    plt.show()

    kernel = np.ones((5, 5), np.uint8)

    # Perform dilation
    # dilate then erod to clear noise
    dilation = cv2.dilate(binary, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    erosion = erosion.astype(np.uint8)

    plt.title("Structuring Element")
    plt.imshow(erosion)
    plt.show()

    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height,width = occupancy_grid.shape
    empty_image = np.zeros((height, width, 3), dtype=np.uint8)
    countoured_image = cv2.drawContours(empty_image, contours, -1, (0, 255, 0), 2)

    save_df = pd.DataFrame(erosion)
    save_df.to_csv(directory+"occupancy_grid_pp.csv")

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)

    
    drawing = np.zeros((countoured_image.shape[0], countoured_image.shape[1], 3), dtype=np.uint8)
    


    f = open(directory+"bounding_boxes.csv", "w")
    f.write("obj,x0,y0,x1,y1,x2,y2,x3,y3")

    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour

        cv2.drawContours(drawing, contours, i, color, thickness=10)
        
        # rotated rectangle
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        print(box)
        cv2.drawContours(drawing, [box], 0, color, thickness=10)
        cv2.putText(drawing, str(i), (box[0,0], box[0,1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        box = (box * resolution) - (map_size/2)

        f.write("\n")
        f.write(f"{i},{box[0,0]},{-box[0,1]},{box[1,0]},{-box[1,1]},{box[2,0]},{-box[2,1]},{box[3,0]},{-box[3,1]}")

    f.close()
    print(drawing.shape)
    
    cv2.imwrite(directory+"detected_objects.png",drawing)
    cv2.imshow('Contours', drawing)
    
    cv2.waitKey(0)




