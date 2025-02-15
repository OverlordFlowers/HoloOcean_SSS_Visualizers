import holoocean
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import collections

from waypoint_gen import *

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
                           "RangeBins": 2000,
                           "Azimuth": 170,
                           "RangeMin": 0.5,
                           "RangeMax": 40,
                           "AddSigma": 0.05,
                           "MultSigma": 0.05,
                           "InitOctreeRange": 10,
                           "ViewRegion": True,
                           "ViewOctree": -1

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


Az = 170
minR = 0.5
maxR = 40
binsR = 2000

t = np.arange(0,1600)
r = np.linspace(-maxR, maxR, binsR)
R, T = np.meshgrid(r, t)
data = np.zeros_like(R)

command = [0, 0, 0, 0, 0, 0]
loc = [0, 0, -50]
rot = [0, 0, 90]
tick = 0

f = open("test_metadata.csv", "w")
f.write("min_range,max_range,num_bins,azimuth")
f.write("\n")
f.write(f"{minR}, {maxR}, {binsR}, {Az}")
f.close()

f = open("test.csv", "w")
f.write("tick,x,y,z,roll,pitch,yaw,")

for i in range(binsR-1):
    f.write(f"bin_{i},")
f.write(f"bin_{binsR-1}")
f.write("\n")

z_const = -50

pts = generate_waypoints_line([-20, 0, z_const], [10, 0, z_const], 0.05)
pts2 = generate_waypoints_line([10, 0, z_const], [10, -10, z_const], 0.05)
pts3 = generate_waypoints_line([10, -10, z_const], [-30, -10, z_const], 0.05)
pts4 = generate_waypoints_line([-30, -10, z_const], [-30, 0, z_const], 0.05)
pts5 = generate_waypoints_line([-30, 0, z_const], [-30, 50, z_const], 0.05)
pts6 = generate_waypoints_line([-30, 50, z_const], [20, 50, z_const], 0.05)
pts7 = generate_waypoints_line([20, 50, z_const], [20, -20, z_const], 0.05)
pts.extend(pts2)
pts.extend(pts3)
pts.extend(pts4)
pts.extend(pts5)
pts.extend(pts6)
pts.extend(pts7)

pts = collections.deque(pts)

with holoocean.make(scenario_cfg=config, start_world=False) as env:
   
   while True:
      state = env.tick()

      if not pts:
         break

      # Get the next waypoint from the queue
      loc, rot = pts.popleft()
      env.agents['auv0'].teleport(loc, rot)

      #print(env.agents['auv0'])
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