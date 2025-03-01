import math 
import numpy as np

def generate_waypoints(waypoints, step_size=0.05):
   pts = generate_waypoints_line(waypoints[0], waypoints[1], step_size=step_size)
   for i in range(2, len(waypoints)):
      pts.extend(generate_waypoints_line(waypoints[i-1], waypoints[i], step_size=step_size))
   return pts

def generate_waypoints_line(start, end, step_size):
    waypoints = []

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    distance = math.sqrt(dx**2 + dy**2 + dz**2)

    yaw = np.rad2deg(math.atan2(dy, dx))

    if distance == 0:
        direction = [0, 0, 0]
    else:
        direction = [dx / distance, dy / distance, dz / distance]
    
    steps = int(distance // step_size)
    
    for i in range(steps + 1):
        loc = [start[j] + i * step_size * direction[j] for j in range(3)]
        rot = [0, 0, yaw]
        waypoints.append((loc, rot))

    if any(abs(waypoints[-1][0][j] - end[j]) > 1e-6 for j in range(3)):
        waypoints.append((end, [0, 0, yaw]))
    
    return waypoints

# NOT TESTED
def generate_waypoints_circle(center, radius, num_points):
    waypoints = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        z = center[2]
        loc = [x, y, z]
       
        rot = [0, 0, theta + math.pi / 2]
        waypoints.append((loc, rot))
    return waypoints