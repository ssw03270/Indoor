import heapq
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D
from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon, box

from vector_utils import get_direction_vector, vector_to_angle, create_rectangle, check_ray_collision

def visualize_room_layout(boxes_parsed_data, folder):    
    # 가구 정보 가져오기
    translations = np.array(boxes_parsed_data['translations'])
    sizes = np.array(boxes_parsed_data['sizes'])
    angles = np.array(boxes_parsed_data['angles'])
    
    # 바닥 평면도 정보 가져오기
    vertices = np.array(boxes_parsed_data['floor_plan_vertices'])
    faces = np.array(boxes_parsed_data['floor_plan_faces'])
    
    # 그림 설정
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 바닥 평면도 그리기 (x와 z 축 사용)
    polygons = []
    for face in faces:
        poly = ShapelyPolygon(vertices[face][:, [0, 2]])
        polygons.append(poly)
    
    # 모든 다각형을 합치고 외곽선만 추출
    union = unary_union(polygons)
    exterior = union.exterior
    
    # 외곽선 그리기
    exterior_coords = list(exterior.coords)
    polygon = Polygon(exterior_coords, fill=False, edgecolor='r')
    ax.add_patch(polygon)
    
    # 축 범위 설정 (x와 z 축 사용)
    x_min, x_max = vertices[:, 0].min() - 1, vertices[:, 0].max() + 1
    z_min, z_max = vertices[:, 2].min() - 1, vertices[:, 2].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)

    center_x = (x_min + x_max) / 2
    center_z = (z_min + z_max) / 2
    
    # 가구 그리기 (x와 z 축 사용)
    for trans, size, angle in zip(translations, sizes, angles):
        width, height = size[0] * 2, size[2] * 2
        
        # 가구의 중심 좌표 계산
        center_furniture_x = trans[0] + center_x
        center_furniture_z = trans[2] + center_z
        
        # 가구의 좌측 하단 좌표 계산
        rect_x = center_furniture_x - width / 2
        rect_z = center_furniture_z - height / 2
        
        rect = Rectangle((rect_x, rect_z), width, height, fill=False, edgecolor='b')
        
        # 회전 변환 생성 (중심 기준)
        t = Affine2D().rotate_around(center_furniture_x, center_furniture_z, -angle[0])
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)
    
    ax.set_aspect('equal')
    ax.set_title(f'Room Layout: {folder}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.invert_yaxis()
    plt.show()

def generate_text_description(scene_type, objects_info, case_count=10):
    """
    방 정보와 오브젝트 정보를 바탕으로 영어 텍스트 설명을 생성합니다.
    
    :param scene_type: 방 유형 (예: 'bedroom', 'diningroom', 'livingroom')
    :param objects_info: 오브젝트 정보 리스트 (각 오브젝트는 'super-category'와 'chatgpt_caption' 키를 포함해야 함)
    :return: 생성된 영어 텍스트 설명
    """
    descriptions = []
    objects_info = objects_info.copy()

    for i in range(case_count):
        description = f"This space is a <{scene_type}>. "
        random.shuffle(objects_info)
        for obj in objects_info:
            description += f"There is a <{obj['category']}> that [{obj['chatgpt_caption']}]. "
        descriptions.append(description)
        
    return descriptions

def random_order_codes(objects):
    random.shuffle(objects)

    generated_code = "<rects>\n"
    for object in objects:
        generated_code += f"  <rect object-category='{object['object-category']}' object-caption='{object['object-caption']}' x='{object['x']}' y='{object['y']}' width='{object['width']}' height='{object['height']}' direction='{object['direction']}' />\n"
    generated_code += "</rects>"
    without_margin_code = generated_code

    # Generate numerical margin code representation
    generated_code = "<rects>\n"
    for object in objects:
        generated_code += f"  <rect object-category='{object['object-category']}' object-caption='{object['object-caption']}' x='{object['x']}' y='{object['y']}' width='{object['width']}' height='{object['height']}' direction='{object['direction']}' margin-top='{object['margin-top'][1]}'  margin-right='{object['margin-right'][1]}'  margin-bottom='{object['margin-bottom'][1]}'  margin-left='{object['margin-left'][1]}' />\n"
    generated_code += "</rects>"
    numerical_margin_code = generated_code

    # Generate discrete margin code representation
    generated_code = "<rects>\n"
    for object in objects:
        generated_code += f"  <rect object-category='{object['object-category']}' object-caption='{object['object-caption']}' x='{object['x']}' y='{object['y']}' width='{object['width']}' height='{object['height']}' direction='{object['direction']}' margin-top='{object['margin-top'][0]}'  margin-right='{object['margin-right'][0]}'  margin-bottom='{object['margin-bottom'][0]}'  margin-left='{object['margin-left'][0]}' />\n"
    generated_code += "</rects>"
    discrete_margin_code = generated_code

    return without_margin_code, numerical_margin_code, discrete_margin_code


def check_overlap(boxes_parsed_data, models_info_parsed_data, folder, debug=True, case_count=10):
    """
    Converts layout data into code representation based on the specified rules.
    
    :param boxes_parsed_data: Parsed box data (dictionary)
    :param folder: Folder name (string)
    :param debug: If True, visualize the layout for debugging purposes
    :return: Code representing the layout (string)
    """
    # Data extraction
    translations = np.array(boxes_parsed_data['translations'])
    sizes = np.array(boxes_parsed_data['sizes'])
    angles = np.array(boxes_parsed_data['angles'])

    categories = []
    captions = []
    for model in models_info_parsed_data:
        categories.append(model['category'])
        captions.append(model['chatgpt_caption'])

    # Discretize angles to 45-degree increments
    angles_deg = np.degrees(angles)
    discrete_angles_deg = np.round(angles_deg / 45) * 45
    discrete_angles = np.radians(discrete_angles_deg)

    vertices = np.array(boxes_parsed_data['floor_plan_vertices'])
    faces = np.array(boxes_parsed_data['floor_plan_faces'])

    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    floor_center_x = (x_min + x_max) / 2
    floor_center_z = (z_min + z_max) / 2

    # Create floor polygons
    floor_polygons = []
    for face in faces:
        poly = ShapelyPolygon(vertices[face][:, [0, 2]] - [floor_center_x, floor_center_z])
        floor_polygons.append(poly)
    
    # Union all polygons and extract exterior boundary
    floor_union = unary_union(floor_polygons)
    floor_exterior = floor_union.exterior

    # Prepare for visualization
    if debug:
        fig, ax = plt.subplots(figsize=(10, 10))
        # Draw floor polygon
        x, y = floor_exterior.xy
        ax.plot(x, y, color='black', linewidth=2, label='Floor')

    # List to store rect data
    rects = []

    # Process each object
    for i, (trans, size, angle) in enumerate(zip(translations, sizes, discrete_angles)):
        width, depth, height = size[0] * 2, size[1] * 2, size[2] * 2

        # Calculate object center coordinates
        cx = trans[0]
        cy = round(trans[1] - depth / 2, 2)
        cz = trans[2]

        # Calculate the coordinates of the object's four corners
        corners = np.array([
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2]
        ])

        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle[0]), -np.sin(angle[0])],
            [np.sin(angle[0]), np.cos(angle[0])]
        ])

        # Rotate and translate corners
        rotated_corners = np.dot(corners, rotation_matrix.T)
        minx, miny, maxx, maxy = ShapelyPolygon(rotated_corners).bounds
        bounding_box = box(minx, miny, maxx, maxy)
        rotated_corners = np.array(bounding_box.exterior.coords)[:-1]

        # Create object polygon
        object_poly = ShapelyPolygon(rotated_corners + [cx, cz])

        # Calculate direction vector
        dx, dz = get_direction_vector(object_poly, floor_exterior, angle, cx, cz, ax=ax if debug else None)
        dx_norm, dz_norm = dx / np.linalg.norm([dx, dz]), dz / np.linalg.norm([dx, dz])

        # Calculate rotation angle to align object to point upwards
        object_angle = vector_to_angle(dx_norm, dz_norm)

        # Get bounding box
        min_x, min_z = rotated_corners.min(axis=0)
        max_x, max_z = rotated_corners.max(axis=0)

        # Calculate rect parameters
        rect_width = max_x - min_x
        rect_height = max_z - min_z
        rect_center_x = (max_x + min_x) / 2 + cx
        rect_center_z = (max_z + min_z) / 2 + cz
        rect_direction = -(object_angle + 270) % 360  # Ensure rotation is between 0 and 360

        # Append rect data
        category = categories[i] if categories else 'unknown'
        caption = captions[i] if captions else 'unknown'
        rects.append({
            'data-category': category,
            'data-caption': caption,
            'x': round(rect_center_x, 2),
            'level': round(cy, 2),
            'y': round(rect_center_z, 2),
            'width': round(rect_width, 2),
            'height': round(rect_height, 2),
            'direction': rect_direction
        })

        if debug:
            # Plot original object
            x, y = (rotated_corners + [cx, cz]).T
            ax.fill(x, y, alpha=0.3, label=f'Object {i+1} Original')

            # Plot direction vector
            ax.arrow(cx, cz, dx_norm * 0.5, dz_norm * 0.5, color='green', width=0.05, head_width=0.2, head_length=0.3)

            # Add index text
            ax.text(cx, cz, f'Object {i+1}, Y: {cy}', fontsize=8, ha='center', va='bottom')

    is_overlap = False
    is_out_of_bound = False
    for idx in range(len(rects)):
        poly1 = create_rectangle(rects[idx]['x'], rects[idx]['y'], rects[idx]['width'] - 0.2, rects[idx]['height'] - 0.2)
        if poly1.intersection(floor_exterior) or rects[idx]['width'] - 0.2 < 0 or rects[idx]['height'] - 0.2 < 0:
            is_out_of_bound = True

        for jdx in range(idx + 1, len(rects)):
            if rects[idx]['level'] == rects[jdx]['level']:
                poly1 = create_rectangle(rects[idx]['x'], rects[idx]['y'], rects[idx]['width'], rects[idx]['height'])
                poly2 = create_rectangle(rects[jdx]['x'], rects[jdx]['y'], rects[jdx]['width'], rects[jdx]['height'])

                # 겹치는 영역 계산
                intersection = poly1.intersection(poly2)
                
                # poly1 기준으로 겹침 비율 계산
                overlap_ratio = intersection.area / poly1.area
                
                # 50% 이상 겹치면 겹침으로 판단
                if overlap_ratio > 0.5:
                    is_overlap = True
                    break
        if is_overlap:
            break

    return is_overlap, is_out_of_bound

def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

# def is_collision(grid, position, radius):
#     x, y = position
#     rows, cols = grid.shape
    
#     for i in range(-radius, radius + 1):
#         for j in range(-radius, radius + 1):
#             nx, ny = x + i, y + j
#             if 0 <= nx < rows and 0 <= ny < cols:
#                 if grid[nx][ny] == 0:
#                     return True  # Collision detected
#             else:
#                 return True  # Out of bounds considered as collision
#     return False  # No collision detected

def a_star_with_radius(grid, start, goal, radius):
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    open_set = set([start])
    
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        open_set.remove(current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if is_collision(grid, neighbor, radius) and heuristic(neighbor, goal) > radius:
                    continue
            else:
                continue
            
            move_cost = np.hypot(i, j)
            tentative_g_score = gscore[current] + move_cost
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue
            
            if neighbor not in open_set or tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                    open_set.add(neighbor)
    
    return False

def is_collision(grid, position, radius):
    x, y = position
    rows, cols = grid.shape
    collision = False
    
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            nx, ny = x + i, y + j
            
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 0:
                    collision = True  # 충돌 감지
                    break
            else:
                collision = True  # 그리드 밖은 충돌로 간주
                break
        if collision:
            break
    return collision

def dijkstra_with_radius(grid, start, goal_points, radius):
    import heapq
    import numpy as np

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    came_from = {}
    gscore = {start: 0}
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    goal_points_set = set(goal_points)
    paths = {}
    
    while open_set and goal_points_set:
        current_gscore, current = heapq.heappop(open_set)
        
        # 현재 위치가 목표 지점 반경 내에 있는지 확인
        reached_goals = [goal for goal in goal_points_set if heuristic(current, goal) <= radius]
        if reached_goals:
            for goal in reached_goals:
                # 경로 재구성
                path = []
                c = current
                while c != start:
                    path.append(c)
                    c = came_from[c]
                path.append(start)
                paths[goal] = path[::-1]
                goal_points_set.remove(goal)
            continue
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if is_collision(grid, neighbor, radius):
                    continue  # 충돌 위치는 건너뜀
            else:
                continue  # 그리드 밖은 무시
                
            move_cost = np.hypot(i, j)
            tentative_g_score = current_gscore + move_cost
            
            if neighbor in gscore and tentative_g_score >= gscore[neighbor]:
                continue
            
            came_from[neighbor] = current
            gscore[neighbor] = tentative_g_score
            heapq.heappush(open_set, (tentative_g_score, neighbor))
    
    return paths