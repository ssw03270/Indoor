import os
import json
import pickle
import numpy as np
from tqdm import tqdm

from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

import matplotlib.pyplot as plt

valid_folder_path = "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes"
folder_paths = ["E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_bedroom", 
                "E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_diningroom", 
                "E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_livingroom"]
file_names = ["threed_front_bedroom_valid_scenes.json", 
              "threed_front_diningroom_valid_scenes.json",
              "threed_front_livingroom_valid_scenes.json"]
rooms = ['bedroom', 'diningroom', 'livingroom']

def visualize_polygons(room_polygon, object_polygons):
    fig, ax = plt.subplots()
    
    # 방 다각형 그리기
    x, y = room_polygon.exterior.xy
    ax.plot(x, y, color='black', linewidth=2)
    
    # 객체 다각형 그리기
    for poly in object_polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y)
    
    minx, miny, maxx, maxy = room_polygon.bounds
    ax.set_xlim(minx - 1, maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)
    ax.set_aspect('equal')
    plt.title('Room Layout with Distance Types')
    plt.show()

def load_polygon(prefix, id, stats, debug=False):
    data = np.load(os.path.join(prefix, id, 'boxes.npz'))

    vertices = np.array(data['floor_plan_vertices'])
    faces = np.array(data['floor_plan_faces'])
    
    # 바닥 평면도 그리기 (x와 z 축 사용)
    polygons = []
    for face in faces:
        poly = Polygon(vertices[face][:, [0, 2]])
        polygons.append(poly)
    
    # 모든 다각형을 합치고 외곽선만 추출
    union = unary_union(polygons)
    room_polygon = Polygon(union.exterior)
    room_centroid = room_polygon.centroid
    room_polygon = translate(room_polygon, xoff=-room_centroid.x, yoff=-room_centroid.y)
    
    room_info = {
        'vertices': vertices.tolist(),
        'faces': faces.tolist(),
        'room_polygon': list(room_polygon.exterior.coords),
        'centroid': [room_centroid.x, room_centroid.y]
    }
    object_infos = []
    object_polygons = []
    for label, size, angle, loc in zip(data['class_labels'], data['sizes'], data['angles'], data['translations']):
        label_idx = np.where(label)[0][0]
        if label_idx >= len(stats['object_types']): # NOTE:
            continue
        cat = stats['object_types'][label_idx]
        
        width, height, depth = size # NOTE: half the actual size
        width, height, depth = width * 2, height * 2, depth * 2
        orientation = round(angle[0] / 3.1415926 * 180)
        dx, dy, dz = loc # NOTE: center point

        # 오브젝트의 좌표를 생성
        rect = Polygon([
            (-width / 2, -depth / 2),
            (width / 2, -depth / 2),
            (width / 2, depth / 2),
            (-width / 2, depth / 2)
        ])
        rotated_rect = rotate(rect, orientation, use_radians=False)
        moved_rect = translate(rotated_rect, xoff=dx, yoff=dz)
        object_infos.append({
            'category': cat,
            'location': [dx, dy, dz],
            'scale': [width / 2, depth / 2, height / 2],
            'rotation': angle[0],
            'polygon': list(moved_rect.exterior.coords),
            'on_floor': True if round(dy - height / 2, 2) < 0.5 else False 
            })
        object_polygons.append(moved_rect)

    if debug:
        visualize_polygons(room_polygon, object_polygons)
    
    return room_info, object_infos

for folder_path, file_name, room in zip(folder_paths, file_names, rooms):
    path = os.path.join(valid_folder_path, file_name)
    with open(path, 'r') as file:
        json_data = json.load(file)
    
    with open(f"{folder_path}/dataset_stats.txt", "r") as file:
        stats = json.load(file)

    room_data = []
    for data in tqdm(json_data):
        scene_id = data['scene_id']
        room_info, object_infos = load_polygon(prefix=folder_path, id=scene_id, stats=stats)

        new_data = {
            'scene_id': scene_id,
            'room_info': room_info,
            'object_infos': data['object_infos'],
            'object_transformations': object_infos
        }
        room_data.append(new_data)
    
    output_file = os.path.join(valid_folder_path, f'{room}_valid_scenes_with_transformation.pkl')
    with open(output_file, 'wb') as pkl_file:
        pickle.dump(room_data, pkl_file)
    print(f"{room}에 대한 정보가 {output_file}에 저장되었습니다.")
        