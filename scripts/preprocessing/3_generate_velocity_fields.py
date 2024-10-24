import os
import pickle
import numpy as np
from tqdm import tqdm

from shapely.geometry import Polygon, Point, LineString
from scipy.interpolate import griddata
from scipy.ndimage import convolve, binary_erosion, gaussian_filter

import matplotlib.pyplot as plt

from utils import a_star_with_radius, dijkstra_with_radius

np.random.seed(42)

# Define global paths and room types
VALID_FOLDER_PATH = "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes"
FOLDER_PATHS = [
    "E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_bedroom",
    "E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_diningroom",
    "E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_livingroom"
]
FILE_NAMES = [
    "threed_front_bedroom_valid_scenes.json",
    "threed_front_diningroom_valid_scenes.json",
    "threed_front_livingroom_valid_scenes.json"
]
ROOMS = ['bedroom', 'diningroom', 'livingroom']


def create_grid_representation(room_polygon, object_polygons, object_height, grid_size=0.1, kernel_size=5):
    """
    Create grid representations of the room and objects.

    Parameters:
        room_polygon (Polygon): Polygon representing the room boundary.
        object_polygons (list of Polygon): List of polygons representing objects in the room.
        grid_size (float): The size of each grid cell.
        kernel_size (int): Kernel size for density map calculation.

    Returns:
        tuple: Contains grids and coordinates for further processing.
    """
    # Get room boundaries
    minx, miny, maxx, maxy = room_polygon.bounds

    # Create grid coordinates
    x = np.arange(minx, maxx, grid_size)
    y = np.arange(miny, maxy, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Flatten grid points
    points = [Point(coord) for coord in zip(xx.flatten(), yy.flatten())]

    # Room grid representation
    room_grid = np.array([room_polygon.contains(point) for point in points]).reshape(xx.shape)

    # Full grid (all ones)
    full_grid = np.ones_like(room_grid)

    # Object grids
    object_grids = []
    object_grids_without_small = []
    object_exterior_grids = []
    for obj_poly, obj_height in zip(object_polygons, object_height):
        obj_grid = np.array([obj_poly.contains(point) for point in points]).reshape(xx.shape)
        object_grids.append(obj_grid)

        # Erode object grid to get interior
        eroded_grid = binary_erosion(obj_grid)

        # Object boundary grid
        boundary_grid = obj_grid & ~eroded_grid
        object_exterior_grids.append(boundary_grid)

    # Combined object grids
    combined_object_grid = np.logical_or.reduce(object_grids) if object_grids else np.zeros_like(room_grid)
    combined_object_exterior_grid = np.logical_or.reduce(object_exterior_grids) if object_exterior_grids else np.zeros_like(room_grid)

    # Empty space grid (inside room and not occupied by objects)
    empty_space_grid = np.logical_or(
        np.logical_and(room_grid, np.logical_not(combined_object_grid)),
        combined_object_exterior_grid
    )
    empty_space_grid_for_density = np.logical_and(full_grid, np.logical_not(combined_object_grid))

    # Calculate density map
    density_map = calculate_density_map(empty_space_grid_for_density, kernel_size=kernel_size)
    density_map[~room_grid] = 1  # Set areas outside the room to 1

    return (
        room_grid,
        combined_object_grid,
        combined_object_exterior_grid,
        object_exterior_grids,
        empty_space_grid,
        density_map,
        xx,
        yy
    )


def calculate_density_map(empty_space_grid, kernel_size=5):
    """
    Calculate the density map for the empty space in the room.

    Parameters:
        empty_space_grid (ndarray): Grid representing empty space.
        kernel_size (int): Size of the convolution kernel.

    Returns:
        ndarray: Density map of the room.
    """
    # Create convolution kernel
    kernel = np.ones((kernel_size, kernel_size))

    # Compute density map via convolution
    density_map = convolve(empty_space_grid.astype(float), kernel, mode='constant', cval=0.0)

    # Normalize density map
    density_map = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map))

    return 1 - density_map  # Invert so low-density areas have higher values


def find_door_location(room_polygon, object_polygons, density_map, xx, yy, door_size=1.0, wall_distance=0.1):
    """
    Find a suitable location for the door in the room.

    Parameters:
        room_polygon (Polygon): Polygon representing the room boundary.
        object_polygons (list of Polygon): List of polygons representing objects in the room.
        density_map (ndarray): Density map of the room.
        xx (ndarray): X-coordinate grid.
        yy (ndarray): Y-coordinate grid.
        door_size (float): Size of the door.
        wall_distance (float): Maximum distance from the wall.

    Returns:
        tuple: Door center coordinates, door polygon, and the closest point on the wall.

    Raises:
        ValueError: If no suitable door location is found.
    """
    boundary = LineString(room_polygon.exterior.coords)

    wall_points = []
    wall_densities = []
    closest_points = []

    min_overlap = float('inf')
    best_wall_point = None
    best_closest_point = None

    for x, y, density in zip(xx.flatten(), yy.flatten(), density_map.flatten()):
        half_size = door_size / 2
        door_candidate = Polygon([
            (x - half_size, y - half_size),
            (x + half_size, y - half_size),
            (x + half_size, y + half_size),
            (x - half_size, y + half_size)
        ])
        if room_polygon.contains(door_candidate) and door_candidate.distance(boundary) <= wall_distance:
            # Closest point on the boundary
            closest_point = boundary.interpolate(boundary.project(door_candidate.centroid))

            # Check for overlap with objects
            is_overlap = any(door_candidate.intersects(obj_poly) for obj_poly in object_polygons)

            if is_overlap:
                # overlap이 발생한 경우, overlap의 양을 계산
                overlap_area = sum([door_candidate.intersection(obj_poly).area for obj_poly in object_polygons])
                if overlap_area < min_overlap:  # 최소 overlap보다 작으면 업데이트
                    min_overlap = overlap_area
                    best_wall_point = (x, y)
                    best_closest_point = closest_point
                continue

            wall_points.append((x, y))
            wall_densities.append(density)
            closest_points.append(closest_point)

    if not wall_points:
        door_center = best_wall_point
        closest_point = best_closest_point

    else:
        # Choose the location with maximum density
        max_density = np.max(wall_densities)
        max_density_indices = np.where(wall_densities == max_density)[0]
        selected_index = np.random.choice(max_density_indices)

        door_center = wall_points[selected_index]
        closest_point = closest_points[selected_index]

    # Create the door polygon
    half_size = door_size / 2
    door_corners = [
        (door_center[0] - half_size, door_center[1] - half_size),
        (door_center[0] + half_size, door_center[1] - half_size),
        (door_center[0] + half_size, door_center[1] + half_size),
        (door_center[0] - half_size, door_center[1] + half_size)
    ]
    door_polygon = Polygon(door_corners)

    return door_center, door_polygon, closest_point


def find_path(door_polygon, empty_space_grid, combined_object_exterior_grid, xx, yy):
    """
    Find paths from the door to all object boundaries using Dijkstra's algorithm.

    Parameters:
        door_polygon (Polygon): Polygon representing the door.
        empty_space_grid (ndarray): Grid representing empty space.
        combined_object_exterior_grid (ndarray): Combined grid of object boundaries.
        xx (ndarray): X-coordinate grid.
        yy (ndarray): Y-coordinate grid.

    Returns:
        list: List of paths to each object's boundary.
    """
    # Create grid points
    points = [Point(coord) for coord in zip(xx.flatten(), yy.flatten())]

    # Door grid representation
    door_grid = np.array([door_polygon.contains(point) for point in points]).reshape(xx.shape)

    # Find start point
    door_points = np.argwhere(door_grid)
    start_point = tuple(np.round(np.mean(door_points, axis=0)).astype(int))  # Convert to integer

    # Goal points (object boundaries)
    goal_points = [tuple(point) for point in np.argwhere(combined_object_exterior_grid).tolist()]

    # Find paths using Dijkstra's algorithm
    paths_dict = dijkstra_with_radius(empty_space_grid, start_point, goal_points, radius=1)
    paths = list(paths_dict.values())

    return paths


def get_velocity_field(room_polygon, grid_map, paths, xx, yy, scale=1):
    """
    Compute the velocity field based on the paths.

    Parameters:
        room_polygon (Polygon): Polygon representing the room boundary.
        grid_map (ndarray): Grid map (0: free space, 1: obstacle).
        paths (list of list of tuples): Paths from the door to each object.
        xx (ndarray): X-coordinate grid.
        yy (ndarray): Y-coordinate grid.
        scale (float): Scaling factor for the velocity vectors.

    Returns:
        tuple: Smoothed U and V components of the velocity field.
    """
    # Initialize velocity field
    velocity_field = np.zeros((grid_map.shape[0], grid_map.shape[1], 2), dtype=float)

    for path in paths:
        for i in range(len(path) - 1):
            current = path[i]
            next_cell = path[i + 1]
            direction = (next_cell[0] - current[0], next_cell[1] - current[1])
            velocity_field[current[0], current[1]] += direction

    # Normalize vectors
    magnitude = np.linalg.norm(velocity_field, axis=2, keepdims=True)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    velocity_field_normalized = velocity_field / magnitude

    # Get U and V components
    U = velocity_field_normalized[:, :, 1]
    V = velocity_field_normalized[:, :, 0]
    U_smooth = gaussian_filter(U, sigma=1)
    V_smooth = gaussian_filter(V, sigma=1)

    return U_smooth, V_smooth


def visualize_room_and_grid_with_particles(
    room,
    scene_id,
    room_polygon,
    object_polygons,
    door_polygon,
    U_smooth,
    V_smooth,
    xx,
    yy,
    num_particles=5000,
    num_steps=10
):
    """
    Visualize the room and simulate particles moving under the influence of the velocity field.

    Parameters:
        scene_id (str): Identifier for the scene.
        room_polygon (Polygon): Polygon representing the room boundary.
        object_polygons (list of Polygon): List of polygons representing objects in the room.
        door_polygon (Polygon): Polygon representing the door.
        U_smooth (ndarray): Smoothed U component of the velocity field.
        V_smooth (ndarray): Smoothed V component of the velocity field.
        xx (ndarray): X-coordinate grid.
        yy (ndarray): Y-coordinate grid.
        num_particles (int): Number of particles to simulate.
        num_steps (int): Number of steps in the simulation.
    """
    # Settings for high-resolution images
    figsize_in_inches = 4
    dpi = 128

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(figsize_in_inches, figsize_in_inches), dpi=dpi, facecolor='black')
    ax.set_facecolor('black')

    # Fill room interior
    x_room, y_room = room_polygon.exterior.xy
    ax.fill(x_room, y_room, color='white')

    # Visualize objects
    for obj in object_polygons:
        x_obj, y_obj = obj.exterior.xy
        ax.fill(x_obj, y_obj, color='gray', alpha=0.5)

    # Visualize door
    if door_polygon:
        x_door, y_door = door_polygon.exterior.xy
        ax.fill(x_door, y_door, color='red')

    # Initialize particle positions within the room
    particle_positions = []
    minx, miny, maxx, maxy = room_polygon.bounds
    while len(particle_positions) < num_particles:
        x_rand = np.random.uniform(minx, maxx)
        y_rand = np.random.uniform(miny, maxy)
        point = Point(x_rand, y_rand)
        if room_polygon.contains(point):
            particle_positions.append([x_rand, y_rand])
    particle_positions = np.array(particle_positions)

    # Create grid points for interpolation
    grid_points = np.array([xx.flatten(), yy.flatten()]).T

    # Simulate particle movement
    for _ in range(num_steps):
        u_interp = griddata(
            grid_points,
            U_smooth.flatten(),
            particle_positions,
            method='linear',
            fill_value=0
        )
        v_interp = griddata(
            grid_points,
            V_smooth.flatten(),
            particle_positions,
            method='linear',
            fill_value=0
        )

        particle_positions[:, 0] += u_interp * 0.01
        particle_positions[:, 1] += v_interp * 0.01

        # Keep particles within room bounds
        particle_positions[:, 0] = np.clip(particle_positions[:, 0], minx, maxx)
        particle_positions[:, 1] = np.clip(particle_positions[:, 1], miny, maxy)

        # Plot particles
        ax.scatter(
            particle_positions[:, 0],
            particle_positions[:, 1],
            color='blue',
            s=0.1,
            alpha=0.3
        )

    # Adjust plot limits
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Equalize axes
    if x_range > y_range:
        pad = (x_range - y_range) / 2
        y_min -= pad
        y_max += pad
    else:
        pad = (y_range - x_range) / 2
        x_min -= pad
        x_max += pad

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Create output directory
    output_path = os.path.join(VALID_FOLDER_PATH, f'{room}_velocity_field')
    os.makedirs(output_path, exist_ok=True)

    # Save figure with particles
    plt.savefig(
        f'{output_path}/{scene_id}_velocity_field.png',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0,
        facecolor=fig.get_facecolor()
    )
    plt.close()

    # Save figure without particles
    fig, ax = plt.subplots(figsize=(figsize_in_inches, figsize_in_inches), dpi=dpi, facecolor='black')
    ax.set_facecolor('black')
    ax.fill(x_room, y_room, color='white')

    # Visualize door
    if door_polygon:
        x_door, y_door = door_polygon.exterior.xy
        ax.fill(x_door, y_door, color='red')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save figure without particles
    plt.savefig(
        f'{output_path}/{scene_id}_condition.png',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0,
        facecolor=fig.get_facecolor()
    )
    plt.close()


def main(room, scene_id, room_polygon, object_polygons, object_height, door_size=1.0):
    """
    Main function to process each scene.

    Parameters:
        scene_id (str): Identifier for the scene.
        room_polygon (Polygon): Polygon representing the room boundary.
        object_polygons (list of Polygon): List of polygons representing objects in the room.
        object_height (list): List of object heights.
        door_size (float): Size of the door.

    Returns:
        tuple: Success flags and output data.
    """
    # Create grid representations
    (
        room_grid,
        combined_object_grid,
        combined_object_exterior_grid,
        object_exterior_grids,
        empty_space_grid,
        density_map,
        xx,
        yy
    ) = create_grid_representation(
        room_polygon,
        object_polygons,
        object_height,
        grid_size=0.1,
        kernel_size=5
    )

    try:
        # Find door location
        door_center, door_polygon, closest_point = find_door_location(
            room_polygon,
            object_polygons,
            density_map,
            xx,
            yy,
            door_size
        )
    except ValueError:
        return False, False, []

    # Find paths
    paths = find_path(door_polygon, empty_space_grid, combined_object_exterior_grid, xx, yy)

    # Check for unreachable objects
    unreachable_objects = []
    for object_index, object_exterior_grid in enumerate(object_exterior_grids):
        able_to_reach = False
        for goal_point in np.argwhere(object_exterior_grid).tolist():
            for path in paths:
                if np.array_equal(goal_point, path[-1]):
                    able_to_reach = True
                    break
            if able_to_reach:
                break
        if not able_to_reach:
            unreachable_objects.append(object_index)

    # Compute velocity field
    U_smooth, V_smooth = get_velocity_field(
        room_polygon,
        empty_space_grid,
        paths,
        xx,
        yy,
        scale=50
    )

    # Visualize results
    visualize_room_and_grid_with_particles(
        room,
        scene_id,
        room_polygon,
        object_polygons,
        door_polygon,
        U_smooth,
        V_smooth,
        xx,
        yy
    )

    return True, len(unreachable_objects) == 0, [
        door_center,
        door_polygon,
        closest_point,
        paths,
        U_smooth,
        V_smooth,
        xx,
        yy
    ]


if __name__ == "__main__":
    for folder_path, file_name, room in zip(FOLDER_PATHS, FILE_NAMES, ROOMS):
        path = os.path.join(VALID_FOLDER_PATH, f'{room}_valid_scenes_with_transformation.pkl')

        door_success_count = 0
        path_success_count = 0
        door_error_count = 0
        path_error_count = 0

        with open(path, 'rb') as file:
            pkl_data = pickle.load(file)
            new_datas = []

            for data in tqdm(pkl_data):
                split = data['split']
                scene_id = data['scene_id']
                room_info = data['room_info']
                object_transformations = data['object_transformations']
                object_infos = data['object_infos']
                room_polygon = Polygon(room_info['room_polygon'])
                object_polygons = [
                    Polygon(obj_trans['polygon'])
                    for obj_trans in object_transformations
                    if 'polygon' in obj_trans and obj_trans['on_floor']
                ]
                object_height = [obj_trans['scale'][1] for obj_trans in object_transformations]

                output_file = os.path.join(VALID_FOLDER_PATH, f'{room}_velocity_field/{scene_id}.pkl')
                # 파일이 이미 존재하는지 확인
                if os.path.exists(output_file):
                    with open(output_file, 'rb') as pkl_file:
                        loaded_data = pickle.load(pkl_file)

                    if split in loaded_data:
                        continue

                    add_new_data = {
                        'split': split,
                        'scene_id': loaded_data['scene_id'],
                        'room_info': loaded_data['room_info'],
                        'object_infos': loaded_data['object_infos'],
                        'object_transformations': loaded_data['object_transformations'],
                        'velocity_field': loaded_data['velocity_field']
                    }
                    
                    # Save the new data
                    output_file = os.path.join(VALID_FOLDER_PATH, f'{room}_velocity_field/{scene_id}.pkl')
                    with open(output_file, 'wb') as pkl_file:
                        pickle.dump(add_new_data, pkl_file)

                    continue  # 파일이 존재하면 다음 반복으로 넘어감

                # Process the scene
                success_door, success_path, outputs = main(
                    room=room,
                    scene_id=scene_id,
                    room_polygon=room_polygon,
                    object_polygons=object_polygons,
                    object_height=object_height,
                    door_size=0.75
                )

                # # Update counts
                # if success_door:
                #     door_success_count += 1
                # else:
                #     door_error_count += 1

                # if success_path:
                #     path_success_count += 1
                # else:
                #     path_error_count += 1

                if success_door:
                    (
                        door_center,
                        door_polygon,
                        closest_point,
                        paths,
                        U_smooth,
                        V_smooth,
                        xx,
                        yy
                    ) = outputs
                    new_data = {
                        'door_success': door_center,
                        'door_polygon': door_polygon,
                        'closest_point': closest_point,
                        'paths': paths,
                        'U_smooth': U_smooth,
                        'V_smooth': V_smooth,
                        'xx': xx,
                        'yy': yy,
                    }
                    add_new_data = {
                        'split': split,
                        'scene_id': scene_id,
                        'room_info': room_info,
                        'object_infos': object_infos,
                        'object_transformations': object_transformations,
                        'velocity_field': new_data
                    }
                    
                    # Save the new data
                    output_file = os.path.join(VALID_FOLDER_PATH, f'{room}_velocity_field/{scene_id}.pkl')
                    with open(output_file, 'wb') as pkl_file:
                        pickle.dump(add_new_data, pkl_file)

            # Calculate success rates
            total_door_count = door_success_count + door_error_count
            success_door_rate = (door_success_count / total_door_count * 100) if total_door_count > 0 else 0
            error_door_rate = (door_error_count / total_door_count * 100) if total_door_count > 0 else 0

            total_path_count = path_success_count + path_error_count
            success_path_rate = (path_success_count / total_path_count * 100) if total_path_count > 0 else 0
            error_path_rate = (path_error_count / total_path_count * 100) if total_path_count > 0 else 0

            # Print statistics
            print(f"Door success count: {door_success_count}, Door failure count: {door_error_count}, "
                  f"Door success rate: {success_door_rate:.2f}%, Door failure rate: {error_door_rate:.2f}%")
            print(f"Path success count: {path_success_count}, Path failure count: {path_error_count}, "
                  f"Path success rate: {success_path_rate:.2f}%, Path failure rate: {error_path_rate:.2f}%")
