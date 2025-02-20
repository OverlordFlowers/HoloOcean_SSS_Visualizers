import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import cv2
import random as rng

def processOccupancyGrid(directory):
    dataframe = pd.read_csv(directory+"occupancy_grid.csv")
    occupancy_grid = dataframe.to_numpy()
    occupancy_grid = occupancy_grid[:,1:]

    # from occupancy grid
    # denoise image (from noisy sonar)
    blurred = gaussian_filter(occupancy_grid, sigma=1)
    plt.imshow(blurred)
    plt.show()

    # threshold to isolate objects # *2 gets rid of noise from sonar
    thresh = (threshold_otsu(blurred)) * 3
    print(thresh)
    binary = blurred > thresh
    binary = binary.astype('float64')
    plt.imshow(binary)
    plt.show()

    kernel = np.ones((5, 5), np.uint8)

    # Perform dilation
    # dilate then erod to clear noise
    dilation = cv2.dilate(binary, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    erosion = erosion.astype(np.uint8)

    plt.imshow(erosion)
    plt.show()

    occupancy_grid *= 255
    occupancy_grid = occupancy_grid.astype(np.uint8)
    masked_image = cv2.bitwise_and(occupancy_grid, occupancy_grid, mask=erosion)
    plt.imshow(masked_image, cmap='copper')
    plt.title("Masked Image")
    plt.show()

    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height,width = masked_image.shape
    empty_image = np.zeros((height, width, 3), dtype=np.uint8)
    countoured_image = cv2.drawContours(empty_image, contours, -1, (0, 255, 0), 2)



    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))



    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv2.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    
    drawing = np.zeros((countoured_image.shape[0], countoured_image.shape[1], 3), dtype=np.uint8)
    
    resolution = 0.1
    map_size = 100

    f = open(directory+"bounding_boxes.csv", "w")
    f.write("obj,x0,y0,x1,y1,x2,y2,x3,y3")

    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour
        cv2.drawContours(drawing, contours, i, color)
        
        # rotated rectangle
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        print(box)
        cv2.drawContours(drawing, [box], 0, color)
        box =  (box * resolution) - (map_size/2)

        f.write("\n")
        f.write(f"{i},{box[0,0]},{box[0,1]},{box[1,0]},{box[1,1]},{box[2,0]},{box[2,1]},{box[3,0]},{box[3,1]}")

    f.close()
    print(drawing.shape)
    cv2.imshow('Contours', drawing)

    countoured_image *= 255
    countoured_image = countoured_image.astype(np.uint8)

    for cx, cy in centroids:
        cv2.circle(countoured_image, (cx, cy), 4, (0, 255, 0), -1)

    print(centroids)

    plt.figure(figsize=(6, 6))
    plt.imshow(countoured_image, cmap='copper')
    plt.title("Detected Objects (Contours)")
    plt.show()

    x_coords, y_coords = zip(*centroids)

    plt.figure(figsize=(6, 6))
    plt.imshow(occupancy_grid, cmap="gray", origin="lower")
    plt.scatter(x_coords, y_coords, color="red", marker="x", label="Centroids")
    plt.title("Sonar Image with Detected Object Centroids")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()
