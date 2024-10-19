import os
import json
import pickle
import numpy as np
from tqdm import tqdm

from shapely.ops import unary_union
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate

import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter

from utils import a_star_with_radius

np.random.seed(42)

valid_folder_path = "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes"
folder_paths = ["E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_bedroom", 
                "E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_diningroom", 
                "E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_livingroom"]
file_names = ["threed_front_bedroom_valid_scenes.json", 
              "threed_front_diningroom_valid_scenes.json",
              "threed_front_livingroom_valid_scenes.json"]
rooms = ['bedroom', 'diningroom', 'livingroom']

def create_grid_representation(room_polygon, object_polygons, grid_size=0.1, kernel_size=5):
    # 방의 경계 구하기
    minx, miny, maxx, maxy = room_polygon.bounds
    
    # 그리드 생성
    x = np.arange(minx, maxx, grid_size)
    y = np.arange(miny, maxy, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # 그리드 포인트 생성
    points = [Point(x, y) for x, y in zip(xx.flatten(), yy.flatten())]

    # 방 그리드 표현
    room_grid = np.array([room_polygon.contains(point) for point in points]).reshape(xx.shape)
    
    # 전체 그리드 생성
    full_grid = np.ones_like(room_grid)
    
    # 객체 그리드 표현
    object_grids = []
    object_exterior_grids = []
    for obj_poly in object_polygons:
        obj_grid = np.array([obj_poly.contains(point) for point in points]).reshape(xx.shape)
        object_grids.append(obj_grid)
        
        # 내부 영역을 침식하여 내부 그리드를 얻음
        eroded_grid = binary_erosion(obj_grid)
        
        # 외곂 그리드는 원래 그리드에서 침식된 그리드를 뺀 것
        boundary_grid = obj_grid & ~eroded_grid
        object_exterior_grids.append(boundary_grid)
    
    # 모든 객체를 합친 그리드 생성
    combined_object_grid = np.logical_or.reduce(object_grids) if object_grids else np.zeros_like(room_grid)
    combined_object_exterior_grid = np.logical_or.reduce(object_exterior_grids) if object_exterior_grids else np.zeros_like(room_grid)
    
    # 빈 공간 그리드 생성 (방 내부이면서 객체가 없는 공간)
    empty_space_grid = np.logical_or(np.logical_and(room_grid, np.logical_not(combined_object_grid)), combined_object_exterior_grid)
    empty_space_grid_for_density = np.logical_and(full_grid, np.logical_not(combined_object_grid))
    
    # 밀도 맵 계산
    density_map = calculate_density_map(empty_space_grid_for_density, kernel_size=kernel_size)
    density_map[~room_grid] = 1  # room_grid를 벗어나는 부분을 0으로 설정
    return room_grid, combined_object_grid, combined_object_exterior_grid, empty_space_grid, density_map, xx, yy

def calculate_density_map(empty_space_grid, kernel_size=5):
    # 커널 생성
    kernel = np.ones((kernel_size, kernel_size))
    
    # 밀도 맵 계산
    density_map = convolve(empty_space_grid.astype(float), kernel, mode='constant', cval=0.0)
    
    # 정규화
    density_map = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map))
    
    return 1 - density_map  # 밀도가 낮은 곳이 높은 값을 가지도록 반전

def find_door_location(room_polygon, object_polygons, density_map, xx, yy, door_size=1.0, wall_distance=0.1):
    boundary = LineString(room_polygon.exterior.coords)
    
    wall_points = []
    wall_densities = []
    
    for x, y, density in zip(xx.flatten(), yy.flatten(), density_map.flatten()):
        half_size = door_size / 2
        door_polygon = Polygon([
            (x - half_size, y - half_size),
            (x + half_size, y - half_size),
            (x + half_size, y + half_size),
            (x - half_size, y + half_size)
        ])
        if room_polygon.contains(door_polygon) and door_polygon.distance(boundary) <= wall_distance:
            # 문 polygon이 방 내부에 완전히 포함되는지 확인
            is_overlap = False
            for object_polygon in object_polygons:
                if door_polygon.intersection(object_polygon):
                    is_overlap = True
                    break

            if is_overlap:
                continue

            wall_points.append((x, y))
            wall_densities.append(density)
    
    if not wall_points:
        raise ValueError("적절한 문 위치를 찾을 수 없습니다. 문 크기를 줄이거나 방 크기를 확인하세요.")
    
    # min_density_index = np.argmin(wall_densities)
    max_density = np.max(wall_densities)
    max_density_indices = np.where(wall_densities == max_density)[0]
    min_density_index = np.random.choice(max_density_indices)

    door_center = wall_points[min_density_index]
    
    half_size = door_size / 2
    door_corners = [
        (door_center[0] - half_size, door_center[1] - half_size),
        (door_center[0] + half_size, door_center[1] - half_size),
        (door_center[0] + half_size, door_center[1] + half_size),
        (door_center[0] - half_size, door_center[1] + half_size)
    ]
    
    door_polygon = Polygon(door_corners)
    
    return door_center, door_polygon

def find_path(door_polygon, empty_space_grid, combined_object_exterior_grid, xx, yy):
    # 그리드 포인트 생성
    points = [Point(x, y) for x, y in zip(xx.flatten(), yy.flatten())]

    # 방 그리드 표현
    door_grid = np.array([door_polygon.contains(point) for point in points]).reshape(xx.shape)

    # door_grid가 True인 곳의 좌표를 찾음
    door_points = np.argwhere(door_grid)
    start_point = tuple(np.round(np.mean(door_points, axis=0)).astype(int))  # 정수로 변환
    goal_points = [tuple(goal_point) for goal_point in np.argwhere(combined_object_exterior_grid).tolist()]  # 튜플로 저장
    
    paths = []
    for goal_point in goal_points:
        path = a_star_with_radius(empty_space_grid, start_point, goal_point, radius=2)
        if path:
            paths.append(path)
    
    return paths

def visualize_room_and_grid(room_polygon, object_polygons, door_size=1.0):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 10))
    
    room_grid, combined_object_grid, combined_object_exterior_grid, empty_space_grid, density_map, xx, yy = create_grid_representation(room_polygon, object_polygons, grid_size=0.1, kernel_size=5)

    try:
        door_center, door_polygon = find_door_location(room_polygon, object_polygons, density_map, xx, yy, door_size)
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.add_patch(plt.Polygon(door_polygon.exterior.coords, fill=False, edgecolor='red', linewidth=2))
            ax.plot(door_center[0], door_center[1], 'r*', markersize=15)
        
        print(f"추정된 문의 중심 위치: {door_center}")
    except ValueError as e:
        print(f"경고: {str(e)}")

    paths = find_path(door_polygon, empty_space_grid, combined_object_exterior_grid, xx, yy)
    
    visualize_polygons(ax1, room_polygon, object_polygons)
    visualize_grid(ax2, room_polygon, room_grid, combined_object_grid)
    visualize_density_map(ax3, room_polygon, density_map)
    visualize_velocity_field(ax4, room_polygon, empty_space_grid, paths, xx, yy, scale=50)

    set_common_axis_limits([ax1, ax2, ax3, ax4], room_polygon)
    
    plt.tight_layout()
    plt.show()

def visualize_polygons(ax, room_polygon, object_polygons):
    x, y = room_polygon.exterior.xy
    ax.plot(x, y, color='black', linewidth=2)
    
    for poly in object_polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y)
    
    ax.set_aspect('equal')
    ax.set_title('실내 레이아웃')

def visualize_grid(ax, room_polygon, room_grid, combined_object_grid):
    minx, miny, maxx, maxy = room_polygon.bounds
    ax.imshow(room_grid, cmap='binary', origin='lower', extent=[minx, maxx, miny, maxy])
    ax.imshow(combined_object_grid, cmap='rainbow', alpha=0.5, origin='lower', extent=[minx, maxx, miny, maxy])
    ax.set_aspect('equal')
    ax.set_title('그리드 표현')

def visualize_empty_grid(ax, room_polygon, empty_grid, combined_object_exterior_grid):
    minx, miny, maxx, maxy = room_polygon.bounds
    ax.imshow(empty_grid, cmap='binary', origin='lower', extent=[minx, maxx, miny, maxy])
    ax.imshow(combined_object_exterior_grid, cmap='rainbow', alpha=0.5, origin='lower', extent=[minx, maxx, miny, maxy])
    ax.set_aspect('equal')
    ax.set_title('그리드 표현')

def visualize_density_map(ax, room_polygon, density_map):
    minx, miny, maxx, maxy = room_polygon.bounds
    # density_plot = ax.imshow(density_map, cmap='YlOrRd', alpha=0.7, origin='lower', extent=[minx, maxx, miny, maxy])
    # plt.colorbar(density_plot, ax=ax, label='오브젝트 밀도')
    ax.imshow(density_map, cmap='YlOrRd', alpha=0.7, origin='lower', extent=[minx, maxx, miny, maxy])
    ax.set_aspect('equal')
    ax.set_title('밀도 맵')

def visualize_velocity_field(ax, room_polygon, grid_map, paths, xx, yy, scale=1, color='red'):
    """
    속도 필드를 계산하고 시각화하는 함수.

    Parameters:
        ax (matplotlib.axes.Axes): 시각화를 그릴 matplotlib 축 객체.
        grid_map (np.ndarray): 그리드 맵 (0: 자유 공간, 1: 장애물).
        paths (list of list of tuples): 각 경로는 (row, col) 튜플의 리스트.
        xx (np.ndarray): X 좌표 배열.
        yy (np.ndarray): Y 좌표 배열.
        scale (float): quiver 벡터의 크기 스케일링 팩터.
        color (str): quiver 벡터의 색상.
    """
    # 속도 필드를 0으로 초기화
    velocity_field = np.zeros((grid_map.shape[0], grid_map.shape[1], 2), dtype=float)

    for path in paths:
        for i in range(len(path)-1):
            current = path[i]
            next_cell = path[i+1]
            direction = (next_cell[0] - current[0], next_cell[1] - current[1])
            velocity_field[current[0], current[1]] += direction

    # 벡터 정규화
    magnitude = np.linalg.norm(velocity_field, axis=2, keepdims=True)
    magnitude[magnitude == 0] = 1  # 0으로 나누는 것을 방지
    velocity_field_normalized = velocity_field / magnitude

    # 속도 필드 시각화
    U = velocity_field_normalized[:,:,1]
    V = velocity_field_normalized[:,:,0]
    U_smooth = gaussian_filter(U, sigma=1)
    V_smooth = gaussian_filter(V, sigma=1)

    minx, miny, maxx, maxy = room_polygon.bounds
    ax.imshow(grid_map, cmap='binary', origin='lower', extent=[minx, maxx, miny, maxy])
    ax.quiver(xx, yy, U_smooth, V_smooth, color=color, scale=scale)
    ax.set_aspect('equal')
    ax.set_title('속도 필드 (두께 고려)')

    print(xx.shape, yy.shape, U_smooth.shape, V_smooth.shape)


def set_common_axis_limits(axes, room_polygon):
    minx, miny, maxx, maxy = room_polygon.bounds
    padding = 1
    for ax in axes:
        ax.set_xlim(minx - padding, maxx + padding)
        ax.set_ylim(miny - padding, maxy + padding)

for folder_path, file_name, room in zip(folder_paths, file_names, rooms):
    path = os.path.join(valid_folder_path, f'{room}_valid_scenes_with_transformation.pkl')
    with open(path, 'rb') as file:
        pkl_data = pickle.load(file)

        for data in pkl_data:
            room_info = data['room_info']
            object_infos = data['object_transformations']

            room_polygon = Polygon(room_info['room_polygon'])
            object_polygons = [Polygon(object_info['polygon']) for object_info in object_infos if 'polygon' in object_info and object_info['on_floor']]

            visualize_room_and_grid(room_polygon=room_polygon, object_polygons=object_polygons, door_size=0.75)