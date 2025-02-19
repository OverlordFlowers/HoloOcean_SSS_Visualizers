from helper_functions.sonarmap_gen import *
from helper_functions.occupancy_grid_processing import *

directory = "outputs/20250219154912/"
#display_sss_map(directory, save_npy=True)
#display_sss_time_map(directory, save_npy=True)
processOccupancyGrid(directory)