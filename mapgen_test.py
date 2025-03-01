from helper_functions.sonarmap_gen import *
from helper_functions.occupancy_grid_processing import *

directory = "outputs/w1/"
display_sss_map(directory, save_npy=True,show_plots=True)
#display_sss_time_map(directory, save_npy=False)
#processOccupancyGrid(directory)