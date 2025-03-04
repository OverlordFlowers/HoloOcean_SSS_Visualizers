# get the scans from waypoints that actually see the objects
# remap the grid to be discrete instead of map coordinates
# optimize the waypoints


import numpy as np
from scipy.spatial.distance import cdist

def compute_waypoint_coverage(waypoints, occupancy_grid, sonar_range):
    coverage = {i: set() for i in range(len(waypoints))}
    grid_height, grid_width = occupancy_grid.shape

    for i, (wx, wy) in enumerate(waypoints):
        gx, gy = int(wx), int(wy)  # Convert to grid indices
        for dx in range(-sonar_range, sonar_range + 1):
            for dy in range(-sonar_range, sonar_range + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    if occupancy_grid[ny, nx] == 1:  # Object pixel
                        coverage[i].add((nx, ny))

    return coverage

def merge_redundant_waypoints(waypoints, coverage):
    unique_waypoints = []
    unique_coverage = {}
    seen_coverages = set()

    for i, wp in enumerate(waypoints):
        coverage_tuple = tuple(sorted(coverage[i]))  # Convert to immutable form
        if coverage_tuple not in seen_coverages:
            seen_coverages.add(coverage_tuple)
            unique_waypoints.append(wp)
            unique_coverage[len(unique_waypoints) - 1] = coverage[i]

    return unique_waypoints, unique_coverage

def tsp_heuristic(waypoints, coverage):
    n = len(waypoints)
    dist_matrix = cdist(waypoints, waypoints)
    unvisited = set(range(n))
    path = [0]  # Start at the first waypoint
    unvisited.remove(0)
    seen_pixels = set(coverage[0])  # Track scanned pixels

    while unvisited:
        last = path[-1]
        best_next = None
        best_score = -np.inf

        for candidate in unvisited:
            travel_cost = dist_matrix[last, candidate]
            new_pixels = coverage[candidate] - seen_pixels
            redundancy_penalty = len(coverage[candidate] & seen_pixels) * 0.5

            score = len(new_pixels) - redundancy_penalty - 0.1 * travel_cost

            if score > best_score:
                best_score = score
                best_next = candidate

        if best_next is None:
            break

        path.append(best_next)
        seen_pixels.update(coverage[best_next])
        unvisited.remove(best_next)

    return [waypoints[i] for i in path]

# Example Data
occupancy_grid = np.random.randint(0, 2, (100, 100))  # Example rasterized map
missions = [
    [(10, 20), (12, 22), (15, 25)],  # Mission 1
    [(30, 40), (32, 42), (35, 45)],  # Mission 2
    [(55, 60), (60, 62)]             # Mission 3
]
sonar_range = 10  # Max sonar detection range

# Step 1: Compute mission-wise waypoint coverage
mission_coverage = [compute_waypoint_coverage(m, occupancy_grid, sonar_range) for m in missions]

# Step 2: Merge redundant waypoints across missions
all_waypoints = [wp for m in missions for wp in m]
all_coverage = {i: cov for m_cov in mission_coverage for i, cov in m_cov.items()}
filtered_waypoints, filtered_coverage = merge_redundant_waypoints(all_waypoints, all_coverage)

# Step 3: Solve TSP with heuristic approach
optimized_path = tsp_heuristic(filtered_waypoints, filtered_coverage)

# Output Results
optimized_path




