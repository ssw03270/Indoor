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


def layout_to_code(boxes_parsed_data, models_info_parsed_data, folder, debug=True, case_count=10):
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
    for idx in range(len(rects)):
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

    distance_type = check_ray_collision(rects, floor_exterior)

    objects = []
    for idx, rect in enumerate(rects):
        objects.append({
            'object-category': rect['data-category'],
            'object-caption': rect['data-caption'],
            'x': rect['x'],
            'y': rect['y'],
            'width': rect['width'],
            'height': rect['height'],
            'direction': rect['direction'],
            'level': rect['level'],
            'margin-top': distance_type[idx]['top'],
            'margin-right': distance_type[idx]['right'],
            'margin-bottom': distance_type[idx]['bottom'],
            'margin-left': distance_type[idx]['left'],
        })

    if debug:
        # Mark the origin
        ax.plot(0, 0, 'ro', markersize=10, label='Origin')

        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Room Layout: {folder}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')
        plt.show()

    # Generate without margin code representation
    without_margin_codes, numerical_margin_codes, discrete_margin_codes = [], [], []
    for i in range(case_count):
        without_margin_code, numerical_margin_code, discrete_margin_code = random_order_codes(objects=objects.copy())
        without_margin_codes.append(without_margin_code)
        numerical_margin_codes.append(numerical_margin_code)
        discrete_margin_codes.append(discrete_margin_code)

    if debug:
        print("마진 없는 코드:")
        print(without_margin_code)
        print("\n수치 마진 코드:")
        print(numerical_margin_code)
        print("\n이산 마진 코드:")
        print(discrete_margin_code)
        print()

    return without_margin_codes, numerical_margin_codes, discrete_margin_codes, is_overlap