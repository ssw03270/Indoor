import numpy as np
import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

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
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()
print(args)

instruction = """
The XZ plane represents the floor, and the Y axis represents the height. 
The width refers to the length extending left and right along the X axis.
The height refers to the length extending up and down along the Y axis. 
The depth refers to the length extending forward and backward along the Z axis. 
Generate a room layout based on the given room requirements.
"""

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

def load_room_boxes(prefix, id, stats, unit):
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

    # if unit in ['px', '']:
    #     condition = f"Condition:\n"
    #     if args.room == 'livingroom':
    #         if 'dining' in id.lower():
    #             condition += f"Room Type: living room & dining room\n"
    #         else:
    #             condition += f"Room Type: living room\n"
    #     else:
    #         condition += f"Room Type: {args.room}\n"
    #     condition += f"Room Size: max length {room_length}{unit}, max width {room_width}{unit}\n"
    # else:
    #     condition = f"Condition:\n" \
    #                 f"Room Type: {args.room}\n" \
    #                 f"Room Size: max length {room_length:.2f}{unit}, max width {room_width:.2f}{unit}\n"
    
    condition = f"<room category={args.room} center-x={x_c:.2f}{unit} center-z={z_c:.2f}{unit} width={room_width:.2f}{unit} depth={room_depth:.2f}{unit}/>\n "

    layouts = []
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

        # if unit in ['px', '']:
        #     layout += f"{cat} {{width: {width}{unit}; " \
        #                         f"depth: {depth}{unit}; " \
        #                         f"height: {height}{unit}; " \
        #                         f"x: {dx}{unit}; " \
        #                         f"y: {dy}{unit}; " \
        #                         f"z: {dz}{unit};" \
        #                         f"orientation: {orientation} degrees;}}\n"                                
        # else:
        #     layout += f"{cat} {{width: {width:.2f}{unit}; " \
        #                         f"depth: {depth:.2f}{unit}; " \
        #                         f"height: {height:.2f}{unit}; " \
        #                         f"orientation: {orientation} degrees; " \
        #                         f"x: {dx:.2f}{unit}; " \
        #                         f"y: {dy:.2f}{unit}; " \
        #                         f"z: {dz:.2f}{unit};}}\n" 
        layout = f"<object category='{cat}' " \
                    f"center-x={dx:.2f}{unit} " \
                    f"center-y={dy:.2f}{unit} " \
                    f"center-z={dz:.2f}{unit} " \
                    f"width={width:.2f}{unit} " \
                    f"height={height:.2f}{unit} " \
                    f"depth={depth:.2f}{unit} " \
                    f"orientation={orientation}degrees/>\n "
        layouts.append(layout)
    
    code = "<objects>\n "
    for layout in layouts:
        code += layout
    code += "</objects>"
    
    return condition, code, dict(data)


def load_set(prefix, ids, stats, unit):
    id2prompt = []
    meta_data = {}
    for id in tqdm(ids):
        input, code, data = load_room_boxes(prefix, id, stats, unit)
        id2prompt.append({"scene_id": id, "instruction": instruction, "input": input, "output": code})
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
    output_dir = f"C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/Preprocessed_{args.room}"
    os.makedirs(output_dir, exist_ok=True)
    # load train examples
    train_ids = splits['rect_train'] if args.regular_floor_plan else splits['train']
    train_data, meta_train_data = load_set(dataset_path, train_ids, stats, args.unit)

        # load val examples
    val_ids = splits['rect_val'] if args.regular_floor_plan else splits['val']
    val_data, meta_val_data = load_set(dataset_path, val_ids, stats, args.unit)
    val_features = load_features(meta_val_data)

        # test 데이터 추가
    test_ids = splits['rect_test'] if args.regular_floor_plan else splits['test']
    test_data, meta_test_data = load_set(dataset_path, test_ids, stats, args.unit)
    test_features = load_features(meta_test_data)
    
    print(f"{len(train_data)}개의 훈련 샘플, {len(val_data)}개의 검증 샘플, {len(test_data)}개의 테스트 샘플을 로드했습니다.")

    if args.icl_type == 'fixed-random':
        # load fixed supporting examples
        all_supporting_examples = list(train_data.values())
        supporting_examples = all_supporting_examples[:args.K]
        train_features = None
    elif args.icl_type == 'k-similar':
        supporting_examples = train_data
        train_features = load_features(meta_train_data)
    
    save_to_json(train_data, f'{args.room}_train_data.json')    
    if train_features is not None:
        save_to_json(train_features, f'{args.room}_train_features.json')

    save_to_json(val_data, f'{args.room}_val_data.json')
    save_to_json(val_features, f'{args.room}_val_features.json')

    save_to_json(test_data, f'{args.room}_test_data.json')
    save_to_json(test_features, f'{args.room}_test_features.json')

    print(f"모든 데이터가 {output_dir}에 JSON 형식으로 저장되었습니다.")