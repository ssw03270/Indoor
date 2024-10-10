import numpy as np
import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog='LayoutGPT for scene synthesis', description='Use GPTs to predict 3D layout for indoor scenes.')
parser.add_argument('--room', type=str, default='bedroom', choices=['bedroom','livingroom'])
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--icl_type', type=str, default='k-similar', choices=['fixed-random', 'k-similar'])
parser.add_argument('--base_output_dir', type=str, default='./llm_output/3D/')
parser.add_argument('--K', type=int, default=8)
parser.add_argument('--unit', type=str, choices=['px', 'm', ''], default='m')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument("--normalize", action='store_true')
parser.add_argument("--regular_floor_plan", action='store_true')
parser.add_argument("--margin_type", type=str, default="discrete", choices=['discrete', 'numerical', 'without'])
args = parser.parse_args()
print(args)

instruction = """
The XZ plane represents the floor, and the Y axis represents the height. 
The width refers to the length extending left and right along the X axis.
The height refers to the length extending up and down along the Y axis. 
The depth refers to the length extending forward and backward along the Z axis. 
Generate a room layout based on the given room requirements.
"""

def create_rectangle(x, y, width, height):
    # 사각형의 반 너비와 반 높이를 구합니다.
    half_width = width / 2
    half_height = height / 2
    
    # 사각형의 네 꼭짓점 좌표를 원점 기준으로 계산합니다.
    corners = [
        (x + -half_width, y + -half_height),
        (x + half_width, y + -half_height),
        (x + half_width, y + half_height),
        (x + -half_width, y + half_height)
    ]
    
    # 다각형을 생성합니다.
    rotated_rectangle = Polygon(corners)
    
    return rotated_rectangle

def check_ray_collision(object_polygons, object_levels, room_polygon, threshold=[[0.1, 'adjacent'], [0.5, 'proximal'], [100, 'distant'], [500, 'unknown']]):
    """
    동일한 레벨의 객체들에 대해 상하좌우로 ray를 쏘고 충돌을 확인합니다.
    
    :param object_polygons: 객체의 Shapely Polygon 리스트
    :param object_levels: 각 객체의 레벨 리스트
    :param room_polygon: 방의 Shapely Polygon
    :param threshold: 충돌 판정을 위한 거리 임계값
    :return: 각 객체의 방향별 거리 타입과 최소 거리
    """
    moe = 0.1   # margin of error
    directions = [(0, 1, 'top'), (0, -1, 'bottom'), (1, 0, 'right'), (-1, 0, 'left')]  # 상, 하, 우, 좌
    distance_list = []
    
    for i, poly1 in enumerate(object_polygons):
        distance_types = {}

        for dx, dy, dir in directions:
            distance_type = None
            min_distance = np.inf
            final_ray = None
            
            # 객체의 중심점 계산
            center = poly1.centroid
            
            for th, _type in threshold:
                # ray 생성
                ray = LineString([center, Point(center.x + dx * 100, center.y + dy * 100)])
                boundary_point = ray.intersection(poly1.exterior)
                ray = LineString([Point(boundary_point.x - dx * moe, boundary_point.y - dy * moe), 
                                  Point(boundary_point.x + dx * th, boundary_point.y + dy * th)])

                for j, poly2 in enumerate(object_polygons):
                    if i != j and object_levels[i] == object_levels[j]:
                        intersection = ray.intersection(poly2)
                        if intersection:
                            distance = boundary_point.distance(intersection)
                            if distance < min_distance:
                                distance_type = _type
                                min_distance = distance
                                final_ray = ray

                # 방 경계와의 충돌 확인
                intersection = ray.intersection(room_polygon.exterior)
                if intersection:
                    distance = boundary_point.distance(intersection)
                    if distance < min_distance:
                        distance_type = _type
                        min_distance = distance
                        final_ray = ray
            
            min_distance = round(min_distance, 2)

            if distance_type is None:
                distance_type = threshold[-1][1]
                min_distance = 100
                final_ray = ray

            distance_types[dir] = [distance_type, min_distance, final_ray]
        distance_list.append(distance_types)

    return distance_list

def save_to_json(data, filename):
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_features(meta_data, floor_plan=True):
    features = {}
    for id, data in tqdm(meta_data.items()):
        if floor_plan:
            features[id] = np.asarray(Image.fromarray(data['room_layout'].squeeze()).resize((64,64))).tolist()
        else:
            room_width = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
            room_depth = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])  
            features[id] = np.asarray([room_width, room_depth]).tolist()
    return features

def visualize_polygons(room_polygon, object_polygons, distance_types):
    fig, ax = plt.subplots()
    
    # 방 다각형 그리기
    x, y = room_polygon.exterior.xy
    ax.plot(x, y, color='black', linewidth=2)
    
    # 객체 다각형 그리기
    for poly in object_polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y)

    # distance type별 색상 설정
    color_map = {
        'adjacent': 'red',
        'proximal': 'blue',
        'distant': 'green',
        'unknown': 'gray'
    }

    # final ray 그리기
    for distance_type in distance_types:
        for direction, (type, distance, ray) in distance_type.items():
            if ray:
                x, y = ray.xy
                ax.plot(x, y, color=color_map[type], linestyle='--', alpha=0.7)
                # 거리 표시
                midpoint = ray.interpolate(0.5)
                ax.text(midpoint.x, midpoint.y, f'{distance:.2f}', 
                        ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    minx, miny, maxx, maxy = room_polygon.bounds
    ax.set_xlim(minx - 1, maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)
    ax.set_aspect('equal')
    plt.legend(handles=[plt.Line2D([0], [0], color=color, label=type) for type, color in color_map.items()])
    plt.title('Room Layout with Distance Types')
    plt.show()

def load_room_boxes(prefix, id, stats, unit, debug=True):
    data = np.load(os.path.join(prefix, id, 'boxes.npz'))

    x_offset  = min(data['floor_plan_vertices'][:,0])
    z_offset = min(data['floor_plan_vertices'][:,2])
    x_c, z_c = data['floor_plan_centroid'][0] - x_offset, data['floor_plan_centroid'][2] - z_offset
    room_width = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
    room_depth = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])    
    vertices = np.stack((data['floor_plan_vertices'][:,0] - x_offset, data['floor_plan_vertices'][:,2] - z_offset), axis=1)
    vertices = np.asarray([list(nxz) for nxz in set(tuple(xz) for xz in vertices)])
    
    # normalize
    if args.normalize:
        norm = min(room_width, room_depth)
        room_width, room_depth = room_width/norm, room_depth/norm
        vertices /= norm
        if unit in ['px', '']:
            scale_factor = 256
            room_width, room_depth = int(room_width*scale_factor), int(room_depth*scale_factor)

    vertices = [f'({v[0]:.2f}, {v[1]:.2f})' for v in vertices]

    condition = f"<room category={args.room} center-x={x_c:.2f}{unit} center-z={z_c:.2f}{unit} width={room_width:.2f}{unit} depth={room_depth:.2f}{unit}/> \n"

    room_polygon = Polygon([
        (x_c - room_width / 2, z_c - room_depth / 2),
        (x_c + room_width / 2, z_c - room_depth / 2),
        (x_c + room_width / 2, z_c + room_depth / 2),
        (x_c - room_width / 2, z_c + room_depth / 2)
    ])

    layouts = []
    object_polygons = []
    object_levels = []
    for label, size, angle, loc in zip(data['class_labels'], data['sizes'], data['angles'], data['translations']):
        label_idx = np.where(label)[0][0]
        if label_idx >= len(stats['object_types']): # NOTE:
            continue
        cat = stats['object_types'][label_idx]
        
        width, height, depth = size # NOTE: half the actual size
        width, height, depth = width * 2, height * 2, depth * 2
        orientation = round(angle[0] / 3.1415926 * 180)
        dx, dy, dz = loc # NOTE: center point
        dx = dx + x_c
        dz = dz + z_c

        # normalize
        if args.normalize:
            width, height, depth = width/norm, height/norm, depth/norm
            dx, dy, dz = dx/norm, dy/norm, dz/norm
            if unit in ['px', '']:
                width, height, depth = int(width*scale_factor), int(height*scale_factor), int(depth*scale_factor)
                dx, dy, dz = int(dx*scale_factor), int(dy*scale_factor), int(dz*scale_factor)

        # 오브젝트의 좌표를 생성
        rect = Polygon([
            (-width / 2, -depth / 2),
            (width / 2, -depth / 2),
            (width / 2, depth / 2),
            (-width / 2, depth / 2)
        ])
        # Shapely로 회전 적용 (degree 단위)
        rotated_rect = rotate(rect, orientation, use_radians=False)
        # Shapely로 이동 적용 (center를 기준으로 이동)
        moved_rect = translate(rotated_rect, xoff=dx, yoff=dz)
        level = round(dy - height / 2, 0)

        object_polygons.append(moved_rect)
        object_levels.append(level)

    distance_types = check_ray_collision(object_polygons, object_levels, room_polygon)

    if debug:
        print(distance_types)
        visualize_polygons(room_polygon, object_polygons, distance_types)

    jids = []
    captions = []
    for label, size, angle, loc, distance_type, jid in zip(data['class_labels'], data['sizes'], data['angles'], data['translations'], distance_types, data['jids']):
        label_idx = np.where(label)[0][0]
        if label_idx >= len(stats['object_types']): # NOTE:
            continue
        cat = stats['object_types'][label_idx]
        
        width, height, depth = size # NOTE: half the actual size
        width, height, depth = width * 2, height * 2, depth * 2
        orientation = round(angle[0] / 3.1415926 * 180)
        dx, dy, dz = loc # NOTE: center point
        dx = dx + x_c
        dz = dz + z_c

        # normalize
        if args.normalize:
            width, height, depth = width/norm, height/norm, depth/norm
            dx, dy, dz = dx/norm, dy/norm, dz/norm
            if unit in ['px', '']:
                width, height, depth = int(width*scale_factor), int(height*scale_factor), int(depth*scale_factor)
                dx, dy, dz = int(dx*scale_factor), int(dy*scale_factor), int(dz*scale_factor)

        
        if args.margin_type == "discrete":
            layout = f"<object category='{cat}' " \
                        f"center-x={dx:.2f}{unit} " \
                        f"center-y={dy:.2f}{unit} " \
                        f"center-z={dz:.2f}{unit} " \
                        f"width={width:.2f}{unit} " \
                        f"height={height:.2f}{unit} " \
                        f"depth={depth:.2f}{unit} " \
                        f"orientation={orientation}degrees " \
                        f"margin-backward={distance_type['top'][0]} " \
                        f"margin-right={distance_type['right'][0]} " \
                        f"margin-forward={distance_type['bottom'][0]} " \
                        f"margin-left={distance_type['left'][0]}/> \n"
            
        elif args.margin_type == "numerical":
            layout = f"<object category='{cat}' " \
                        f"center-x={dx:.2f}{unit} " \
                        f"center-y={dy:.2f}{unit} " \
                        f"center-z={dz:.2f}{unit} " \
                        f"width={width:.2f}{unit} " \
                        f"height={height:.2f}{unit} " \
                        f"depth={depth:.2f}{unit} " \
                        f"orientation={orientation}degrees " \
                        f"margin-backward={distance_type['top'][1]}{unit} " \
                        f"margin-right={distance_type['right'][1]}{unit} " \
                        f"margin-forward={distance_type['bottom'][1]}{unit} " \
                        f"margin-left={distance_type['left'][1]}{unit}/> \n"
            
        elif args.margin_type == "without":
            layout = f"<object category='{cat}' " \
                        f"center-x={dx:.2f}{unit} " \
                        f"center-y={dy:.2f}{unit} " \
                        f"center-z={dz:.2f}{unit} " \
                        f"width={width:.2f}{unit} " \
                        f"height={height:.2f}{unit} " \
                        f"depth={depth:.2f}{unit} " \
                        f"orientation={orientation}degrees "

        layouts.append(layout)
        jids.append(jid)

        caption_path = os.path.join("C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/InstructScene/3D-FUTURE-chatgpt", f"{jid}.txt")
        try:
            with open(caption_path, 'r', encoding='utf-8') as caption_file:
                caption = caption_file.read().strip()
        except FileNotFoundError:
            caption = f"{cat}"
        captions.append(caption)
    
    code = "<objects> \n"
    for layout in layouts:
        code += layout
    code += "</objects>"
    
    return condition, code, dict(data), jids, captions


def load_set(prefix, ids, stats, unit):
    id2prompt = []
    meta_data = {}
    for id in tqdm(ids):
        input, code, data, jids, captions = load_room_boxes(prefix, id, stats, unit)
        id2prompt.append({"scene_id": id, "object_ids": jids, "object_captions": captions, "instruction": instruction, "input": input, "output": code})
        meta_data[id] = data
    return id2prompt, meta_data

if __name__ == '__main__':
    if args.room == "bedroom":
        data_split_path = "C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/bedroom_splits.json"
        dataset_path = "C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/data_output/bedroom"

    elif args.room == "livingroom":
        data_split_path = "C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/livingroom_splits.json"
        dataset_path = "C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/data_output/livingroom"

    with open(data_split_path, "r") as file:
        splits = json.load(file)
        
    with open(f"{dataset_path}/dataset_stats.txt", "r") as file:
        stats = json.load(file)

    # check if have been processed
    output_dir = f"C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/Preprocessed_{args.room}/{args.margin_type}_margin"
    os.makedirs(output_dir, exist_ok=True)
    # load train examples
    train_ids = splits['rect_train'] if args.regular_floor_plan else splits['train']
    train_data, meta_train_data = load_set(dataset_path, train_ids, stats, args.unit)

        # load val examples
    val_ids = splits['rect_val'] if args.regular_floor_plan else splits['val']
    val_data, meta_val_data = load_set(dataset_path, val_ids, stats, args.unit)
    # val_features = load_features(meta_val_data)

        # test 데이터 추가
    test_ids = splits['rect_test'] if args.regular_floor_plan else splits['test']
    test_data, meta_test_data = load_set(dataset_path, test_ids, stats, args.unit)
    # test_features = load_features(meta_test_data)
    
    print(f"{len(train_data)}개의 훈련 샘플, {len(val_data)}개의 검증 샘플, {len(test_data)}개의 테스트 샘플을 로드했습니다.")

    if args.icl_type == 'fixed-random':
        # load fixed supporting examples
        all_supporting_examples = list(train_data.values())
        supporting_examples = all_supporting_examples[:args.K]
        # train_features = None
    elif args.icl_type == 'k-similar':
        supporting_examples = train_data
        # train_features = load_features(meta_train_data)
    
    save_to_json(train_data, f'{args.room}_train_data.json')    
    # if train_features is not None:
    #     save_to_json(train_features, f'{args.room}_train_features.json')

    save_to_json(val_data, f'{args.room}_val_data.json')
    # save_to_json(val_features, f'{args.room}_val_features.json')

    save_to_json(test_data, f'{args.room}_test_data.json')
    # save_to_json(test_features, f'{args.room}_test_features.json')

    print(f"모든 데이터가 {output_dir}에 JSON 형식으로 저장되었습니다.")


