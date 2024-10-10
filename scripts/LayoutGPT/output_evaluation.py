import json
import re
import numpy as np

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# Parse rect data from XML
# def parse_rect_data(xml_string):
#     rects = []
#     pattern = r"<object\s+category='([^']+)'\s+center-x=(\d+\.?\d*)m\s+center-y=(\d+\.?\d*)m\s+center-z=(\d+\.?\d*)m\s+width=(\d+\.?\d*)m\s+height=(\d+\.?\d*)m\s+depth=(\d+\.?\d*)m\s+orientation=(-?\d+\.?\d*)degrees/>"
    
#     matches = re.findall(pattern, xml_string)
    
#     for match in matches:
#         rect = {
#             'category': match[0],
#             'x': float(match[1]),
#             'y': float(match[2]),
#             'z': float(match[3]),
#             'width': float(match[4]),
#             'height': float(match[5]),
#             'depth': float(match[6]),
#             'orientation': float(match[7])
#         }
#         rects.append(rect)
#     return rects
    
# def parse_rect_data(xml_string):
#     rects = []
#     pattern = r"<object\s+category='([^']+)'\s+center-x=(\d+\.?\d*)m\s+center-y=(\d+\.?\d*)m\s+center-z=(\d+\.?\d*)m\s+width=(\d+\.?\d*)m\s+height=(\d+\.?\d*)m\s+depth=(\d+\.?\d*)m\s+orientation=(-?\d+\.?\d*)degrees\s+margin-backward=(\w+)\s+margin-right=(\w+)\s+margin-forward=(\w+)\s+margin-left=(\w+)/>"
    
#     matches = re.findall(pattern, xml_string)
    
#     for match in matches:
#         rect = {
#             'category': match[0],
#             'x': float(match[1]),
#             'y': float(match[2]),
#             'z': float(match[3]),
#             'width': float(match[4]),
#             'height': float(match[5]),
#             'depth': float(match[6]),
#             'orientation': float(match[7]),
#             'margin_backward': match[8],
#             'margin_right': match[9],
#             'margin_forward': match[10],
#             'margin_left': match[11]
#         }
#         rects.append(rect)
#     return rects
def parse_rect_data(xml_string):
    rects = []
    pattern = r"<object\s+category='([^']+)'\s+center-x=(\d+\.?\d*)m\s+center-y=(\d+\.?\d*)m\s+center-z=(\d+\.?\d*)m\s+width=(\d+\.?\d*)m\s+height=(\d+\.?\d*)m\s+depth=(\d+\.?\d*)m\s+orientation=(-?\d+\.?\d*)degrees\s+margin-backward=(\d+\.?\d*)m\s+margin-right=(\d+\.?\d*)m\s+margin-forward=(\d+\.?\d*)m\s+margin-left=(\d+\.?\d*)m/>"
    
    matches = re.findall(pattern, xml_string)
    
    for match in matches:
        rect = {
            'category': match[0],
            'x': float(match[1]),
            'y': float(match[2]),
            'z': float(match[3]),
            'width': float(match[4]),
            'height': float(match[5]),
            'depth': float(match[6]),
            'orientation': float(match[7]),
            'margin_backward': float(match[8]),
            'margin_right': float(match[9]),
            'margin_forward': float(match[10]),
            'margin_left': float(match[11])
        }
        rects.append(rect)
    return rects

# Parse room data from XML
def parse_room_data(xml_string):
    pattern = r"<room\s+category=(\w+)\s+center-x=(-?\d+\.?\d*)m\s+center-z=(-?\d+\.?\d*)m\s+width=(\d+\.?\d*)m\s+depth=(\d+\.?\d*)m/>"
    match = re.search(pattern, xml_string)
    
    if match:
        room = {
            'category': match.group(1),
            'center_x': float(match.group(2)),
            'center_z': float(match.group(3)),
            'width': float(match.group(4)),
            'depth': float(match.group(5))
        }
        return room

def parse_example(example_string):
    # 방 정보 파싱
    room = parse_room_data(example_string)
    
    # 객체 정보 파싱
    objects = parse_rect_data(example_string)
    
    return room, objects

def invalid_object_size(room_data, object_data, margin=0.1):
    invalid_objects = []
    
    room_polygon = Polygon([
        (room_data['center_x'] - room_data['width'] / 2 - margin, room_data['center_z'] - room_data['depth'] / 2 - margin),
        (room_data['center_x'] + room_data['width'] / 2 + margin, room_data['center_z'] - room_data['depth'] / 2 - margin),
        (room_data['center_x'] + room_data['width'] / 2 + margin, room_data['center_z'] + room_data['depth'] / 2 + margin),
        (room_data['center_x'] - room_data['width'] / 2 - margin, room_data['center_z'] + room_data['depth'] / 2 + margin)
    ])

    for obj in object_data:
        # 오브젝트의 좌표를 생성
        object_polygon = Polygon([
            (-obj['width'] / 2, -obj['depth'] / 2),
            (obj['width'] / 2, -obj['depth'] / 2),
            (obj['width'] / 2, obj['depth'] / 2),
            (-obj['width'] / 2, obj['depth'] / 2)
        ])

        # Shapely로 회전 적용 (degree 단위)
        object_polygon = rotate(object_polygon, obj['orientation'], use_radians=False)

        # Shapely로 이동 적용 (center를 기준으로 이동)
        object_polygon = translate(object_polygon, xoff=obj['x'], yoff=obj['z'])
        # 오브젝트가 방 밖에 있는지 검사
        if not room_polygon.contains(object_polygon):
            invalid_objects.append({
                'category': obj['category'],
                'reason': '방 밖에 위치함'
            })
    
    return invalid_objects

def categorical_kl(p, q):
        return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()

def object_category_KL_divergence(output, gt_data):
    with open(f"C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/data_output/bedroom/dataset_stats.txt", "r") as file:
        stats = json.load(file)

    all_categories = stats['object_types']
    pred_label_freq = {c: 0 for c in all_categories}
    gt_label_freq = {c: 0 for c in all_categories}

    for data in output:
        for object_data in data:
            category = object_data['category']
            if category not in pred_label_freq:
                continue
            pred_label_freq[category] += 1
    pred_label_freq = np.asarray([pred_label_freq[k]/sum(list(pred_label_freq.values())) for k in sorted(all_categories)])
    
    for data in gt_data:
        for object_data in data:
            category = object_data['category']
            if category not in gt_label_freq:
                continue
            gt_label_freq[category] += 1
    gt_label_freq = np.asarray([gt_label_freq[k]/sum(list(gt_label_freq.values())) for k in sorted(all_categories)])
    
    kl_div = categorical_kl(gt_label_freq, pred_label_freq)

    return kl_div

if __name__ == '__main__':
    gt_file_path = "C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/Preprocessed_bedroom/numerical_margin/bedroom_test_data.json"
    output_file_path = "C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/Outputs/numerical_margin_outputs/inference_results.json"
    with open(gt_file_path, "r") as file:
        gt_data = json.load(file)
    with open(output_file_path, "r", encoding="utf-8") as file:
        output_data = json.load(file)

    gt_invalid_scenes = []
    gt_objects_list = []
    for data in gt_data:
        rects = parse_rect_data(data['output'])
        room = parse_room_data(data['input'])
        invalid_objects = invalid_object_size(room, rects)
        
        if len(invalid_objects) > 0:
            gt_invalid_scenes.append(data['scene_id'])
        
        gt_objects_list.append(rects)

    print("GT Out-of-bound rate: ", round(len(gt_invalid_scenes) / len(output_data) * 100, 2))

    output_invalid_scenes = []
    output_objects_list = []
    for data in output_data:
        room, rects = parse_example(data['output'])
        invalid_objects = invalid_object_size(room, rects)
        
        if len(invalid_objects) > 0:
            output_invalid_scenes.append(data['input'])
        
        output_objects_list.append(rects)

    print("Output Out-of-bound rate: ", round(len(output_invalid_scenes) / len(output_data) * 100, 2))

    KL_divergence = object_category_KL_divergence(output=output_objects_list, gt_data=gt_objects_list)
    print("KL Divergence: ", KL_divergence)