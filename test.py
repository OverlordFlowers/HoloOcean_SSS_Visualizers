import holoocean
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import collections
from datetime import datetime
import os
import csv

from helper_functions.sonarmap_gen import *
from helper_functions.waypoint_gen import *
#from helper_functions.waypoint_optimization import *
from helper_functions.occupancy_grid_processing import *

timestamp = datetime.today().strftime("%Y%m%d%H%M%S")

# SSS Parameters
Az = 170
minR = 0.5
maxR = 10
binsR = 2000

z_const = -50
map_size = 100   # in meters
resolution = 0.1 # meters per pixel

config = {
      "name": "test",
      "world": "ExampleLevel",
      "main_agent": "auv0",
      
      "agents": [
            {
               "agent_name": "auv0",
               "agent_type": "TorpedoAUV",
               "sensors": [
                  {
                        "sensor_type": "LocationSensor",
                  },
                  {
                     "sensor_type": "DepthSensor",
                     "socket": "DepthSocket",
                     "Hz": 10,
                     "configuration": {
                           "Sigma": 0.255
                     }
                  },
                  {
                     "sensor_type": "SidescanSonar",
                     "socket": "SonarSocket",
                     "Hz": 30,
                     "configuration": {
                           "RangeBins": binsR,
                           "Azimuth": Az,
                           "RangeMin": minR,
                           "RangeMax": maxR,
                           "AddSigma": 0.05,
                           "MultSigma": 0.05,
                           "InitOctreeRange": 10
                           #"ViewRegion": True,
                           #"ViewOctree": -1

                     }
                  },
                  {
                     "sensor_type": "DynamicsSensor",
                     "socket": "COM",
                  }
               ],
               "control_scheme": 1,
               "location": [50, 50, 2]
            }
      ]
   }

marker = 't1'


m1_waypoints = [[-20, 0, z_const],
             [10, 0, z_const],
             [10, -10, z_const],
             [-30, -10, z_const],
             [-30, 50, z_const],
             [20, 50, z_const],
             [20, -20, z_const],
             [-30, 30, z_const]]


m2_waypoints = [[-40, 40, z_const],
                [20, 40, z_const],
                [20, 30, z_const],
                [-40, 30, z_const],
                [-40, 20, z_const],
                [20, 20, z_const],
                [20, 10, z_const]]



m3_waypoints = [[20, 40, z_const],
                [20, -40, z_const],
                [0, -40, z_const],
                [0, 40, z_const],
                [-20, 40, z_const],
                [-20, -40, z_const]]


m4_waypoints = [[-20, -20, z_const],
                [0, -20, z_const],
                [-30, 20, z_const],
                [0, 40, z_const],
                [20, 0, z_const]]


comprehensive = [[-50, -50, z_const],
                 [-50, 50, z_const],
                 [-40, 50, z_const],
                 [-40, -50, z_const],
                 [-30, -50, z_const],
                 [-30, 50, z_const],
                 [-20, 50, z_const],
                 [-20, -50, z_const],
                 [-10, -50, z_const],
                 [-10, 50, z_const],
                 [0, 50, z_const],
                 [0, -50, z_const],
                 [10, -50, z_const],
                 [10, 50, z_const],
                 [20, 50, z_const],
                 [20, -50, z_const],
                 [30, -50, z_const],
                 [30, 50, z_const],
                 [40, 50, z_const],
                 [40, -50, z_const],
                  [50, -50, z_const],
                 [50, 50, z_const]]
   

# Create parent folder if it doesn't exist
directory_name = "outputs"
try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")

if (marker is None):
   directory_name = "outputs/"+timestamp
else:
   directory_name = "outputs/"+marker

try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")

directory = directory_name+"/"

def write_metadata(directory, minR, maxR, binsR, Az, mapsize, resolution, marker=None):
   if marker is None:
      f = open(directory+"metadata.csv", "w")
   else:
      f = open(marker+"_"+timestamp+"metadata.csv", "w")

   f.write("min_range,max_range,num_bins,azimuth,mapsize,resolution")
   f.write("\n")
   f.write(f"{minR},{maxR},{binsR},{Az},{mapsize},{resolution}")
   f.close()

write_metadata(directory, minR, maxR, binsR, Az, map_size, resolution)
'''
with open(directory+"waypoints.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["x", "y", "z"])
    writer.writerows(m1_waypoints)
'''
'''
df = pd.read_csv("optimized_waypoint.csv")
pts = df.to_numpy()[:,:]
print(pts)
'''
with holoocean.make(scenario_cfg=config, start_world=False) as env:
   
   pts = generate_waypoints(comprehensive)
   pts_array = np.array(pts)
   pts = collections.deque(pts)

   tick = 0

   t = np.arange(0,1600)
   r = np.linspace(-maxR, maxR, binsR)
   R, T = np.meshgrid(r, t)
   data = np.zeros_like(R)

   f = open(directory+"scan.csv", "w")
   f.write("tick,x,y,z,roll,pitch,yaw,")

   for i in range(binsR-1):
      f.write(f"bin_{i},")
   f.write(f"bin_{binsR-1}")
   f.write("\n")

   print("Beginning Simulation")
   while True:
      state = env.tick()
      '''
      if not pts:
         break
      '''
      # Get the next waypoint from the queue
      loc = pts.popleft()
      #loc[1] = loc[1]
      #env.agents['auv0'].teleport(loc[0:3], [0, 0, loc[3]])
      #env.agents['auv0'].teleport(loc[0], loc[1])

      if 'SidescanSonar' in state:
         data = np.roll(data, 1, axis=0)
         data[0] = state['SidescanSonar']

      if 'DynamicsSensor' in state:
         dynamics = state['DynamicsSensor']
         
         curr_pos = dynamics[6:9]
         curr_rpy = dynamics[15:18]
         print(f"{curr_pos}, {curr_rpy}")
      
      sonar_scan = ', '.join([str(x) for x in data[0]])
      curr_scan = f"{tick}, {curr_pos[0]}, {curr_pos[1]}, {curr_pos[2]}, {curr_rpy[0]}, {curr_rpy[1]}, {curr_rpy[2]}, "
      curr_scan = curr_scan+sonar_scan
      f.write(curr_scan)
      f.write("\n")
      tick = tick + 1
      

f.close()
print("Job's done!")

display_sss_map(directory, map_size=map_size, resolution=resolution, save_npy=True, show_plots=True)
display_sss_time_map(directory, save_npy=True, show_plots=True)
processOccupancyGrid(directory)



      


