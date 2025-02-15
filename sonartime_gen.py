import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

def load_sonar_data(metadata_filename, telemetry_filename):
    metadata = pd.read_csv(metadata_filename)
    min_range = int(metadata['min_range'].iloc[0]) # Something something deprecation
    max_range = int(metadata['max_range'].iloc[0])
    num_bins = int(metadata['num_bins'].iloc[0])
    azimuth = int(metadata['azimuth'].iloc[0]) # Unused for now, could prove useful in the future

    telemetry = pd.read_csv(telemetry_filename)

    scans = (telemetry.to_numpy())

    return min_range, max_range, num_bins, scans

def gen_sonarplot(min_range, max_range, num_bins, scans, window_size=2400):
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
        
def display_sss_time_map(metadata_filename, telemetry_filename, window_size=2400):
    start_time = time.time()
    min_range, max_range, num_bins, scans = load_sonar_data(metadata_filename, telemetry_filename)
    end_time = time.time()
    print(f"Time to load data: {end_time - start_time} seconds")
    
    gen_sonarplot(min_range, max_range, num_bins, scans, window_size)

