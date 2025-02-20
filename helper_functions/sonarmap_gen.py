import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

def load_sonar_data(directory):
    metadata = pd.read_csv(directory+"metadata.csv")
    min_range = int(metadata['min_range'].iloc[0]) # Something something deprecation
    max_range = int(metadata['max_range'].iloc[0])
    num_bins = int(metadata['num_bins'].iloc[0])
    azimuth = int(metadata['azimuth'].iloc[0]) # Unused for now, could prove useful in the future

    telemetry = pd.read_csv(directory+"scan.csv")

    scans = (telemetry.to_numpy())

    return min_range, max_range, num_bins, scans

def generate_side_scan_map(directory, min_range, max_range, num_bins, scans, map_size=100, resolution=0.1, save_npy=False):
    """
    Creates a simple occupancy grid map from side-scan sonar scans.
    
    Parameters:
      - min_range, max_range, num_bins: sonar sensor parameters.
      - scans: list of sonar scans (with pose and sonar returns).
      - map_size: size (in meters) of the square map (centered at (0,0)).
      - resolution: size (in meters) of each grid cell.
    
    Assumes:
      - num_bins is even.
      - The left side returns (bins 0 to num_bins/2 - 1) are at an angle of yaw + 90°.
      - The right side returns (bins num_bins/2 to num_bins - 1) are at an angle of yaw - 90°.
      - Each side’s range resolution is computed based on half the number of bins.
    
    Returns:
      - occupancy_grid: 2D numpy array with accumulated sonar intensity.
    """
    grid_size = int(map_size / resolution)
    occupancy_grid = np.zeros((grid_size, grid_size))
    bins_per_side = num_bins // 2
    counter = 0
    for row in scans:
        #counter = counter + 1
        if (counter > 1000):
            break
        sensor_x = row[1]
        sensor_y = row[2]
        yaw = row[6]
        sonar_returns = row[7:]
        
        # port
        for i in range(bins_per_side):
            intensity = sonar_returns[bins_per_side-1-i]

            r = min_range + (i + 0.5) * (max_range - min_range) / bins_per_side

            angle = np.deg2rad(yaw) + math.pi/2
            point_x = sensor_x + r * math.cos(angle)
            point_y = (sensor_y) + r * math.sin(angle)

            grid_x = int((point_x + map_size/2) / resolution)
            grid_y = int((point_y + map_size/2) / resolution)
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:

                occupancy_grid[grid_y, grid_x] = max(occupancy_grid[grid_y, grid_x], intensity)
        
        # starboard
        for i in range(bins_per_side, num_bins):
            intensity = sonar_returns[i]

            j = i - bins_per_side
            r = min_range + (j + 0.5) * (max_range - min_range) / bins_per_side

            angle = np.deg2rad(yaw) - math.pi/2
            point_x = sensor_x + r * math.cos(angle)
            point_y = (sensor_y) + r * math.sin(angle)
            
            grid_x = int((point_x + map_size/2) / resolution)
            grid_y = int((point_y + map_size/2) / resolution)
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                occupancy_grid[grid_y, grid_x] = max(occupancy_grid[grid_y, grid_x], intensity)

    save_df = pd.DataFrame(occupancy_grid)
    save_df.to_csv(directory+"occupancy_grid.csv")

    return occupancy_grid

def display_sss_map(directory, map_size=100, save_npy=False, show_plots=False):
    total_start_time = time.time()
    start_time = time.time()
    min_range, max_range, num_bins, scans = load_sonar_data(directory)
    end_time = time.time()
    print(f"Time to load sonar data: {end_time - start_time} seconds")

    start_time = time.time()
    occupancy_grid = generate_side_scan_map(directory, min_range, max_range, num_bins, scans, save_npy=save_npy)
    end_time = time.time()
    print(f"Time to generate map: {end_time - start_time} seconds")

    start_time = time.time()
    map_size = 100

    if (show_plots):
      fig, ax = plt.subplots(figsize=(8, 8))

      cax = ax.imshow(occupancy_grid, origin='lower', cmap='gray',
                      extent=[-map_size/2, map_size/2, -map_size/2, map_size/2])
      
      plt.savefig(directory+"occupancy_grid")

      ax.set_title("SSS Map")
      ax.set_xlabel("X (meters)")
      ax.set_ylabel("Y (meters)")
      fig.colorbar(cax, ax=ax, label="Sonar Return Intensity")

      x_pos = scans[0::10,1]
      y_pos = scans[0::10,2]

      times = np.arange(x_pos.shape[0])

      sc = ax.scatter(x_pos, y_pos, c=times, cmap='viridis', s=50, edgecolor='k')
      fig.colorbar(sc, ax=ax, label="Scan Number")
      plt.savefig(directory+"occupancy_grid_wps")
      end_time = time.time()
      print(f"Time to map path: {end_time - start_time} seconds")

      total_end_time = time.time()
      print(f"Total time to display sonar map: {total_end_time - total_start_time} seconds")
      
      plt.show()

def gen_sonarplot(directory, min_range, max_range, num_bins, scans, window_size=2400, save_npy=False, show_plots=False):

    if (save_npy):
        df = pd.DataFrame(scans[:,7:])


    if (show_plots):
        start_time = time.time()

        ticks = scans.shape[0]
        t = np.arange(0,ticks)
        r = np.linspace(-max_range, max_range, num_bins)
        R, T = np.meshgrid(r, t)
        data = np.zeros_like(R)

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.grid(False)
        plot = plt.pcolormesh(R, T, data, cmap='copper', shading='auto', vmin=0, vmax=1)
        plt.tight_layout()
        plot.set_array((scans[:, 7:]))

        plt.savefig(directory+"sonarmap")
        pd.DataFrame(scans[:, 7:]).to_csv(directory+"sonarmap.csv")

        ax.invert_yaxis()
        
        # Set initial y-axis limits to show only a window of ticks.
        # Because the y-axis is inverted, the higher tick number is at the bottom.
        ax.set_ylim(window_size - 0.5, -0.5)
        
        # Create a slider to act as a scrollbar.
        # The slider will control the starting tick of the window.
        slider_ax = fig.add_axes([0.15, 0.01, 0.7, 0.03])
        slider = Slider(slider_ax, 'Tick', 0, ticks - window_size, valinit=0, valstep=1)
        
        def update(val):
            start = int(slider.val)
            end = start + window_size
            ax.set_ylim(end - 0.5, start - 0.5)
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        end_time = time.time()
        print(f"Time to generate map: {end_time - start_time} seconds")
        plt.show()
        
def display_sss_time_map(directory, window_size=2400, save_npy=False, show_plots=False):
    start_time = time.time()
    min_range, max_range, num_bins, scans = load_sonar_data(directory)
    end_time = time.time()
    print(f"Time to load data: {end_time - start_time} seconds")
    
    gen_sonarplot(directory, min_range, max_range, num_bins, scans, window_size, save_npy=save_npy, show_plots=show_plots)