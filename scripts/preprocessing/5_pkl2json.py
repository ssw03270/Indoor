import os
import pickle
import numpy as np
import json
import math

np.random.seed(42)

# Define global paths and room types
VALID_FOLDER_PATH = "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes"
bedroom_class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe", "none"]
diningroom_class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "none"]
livingroom_class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "none"]

class_label_list = [bedroom_class_labels, diningroom_class_labels, livingroom_class_labels]
max_obj_count_list = [12, 21, 21]

def radian_to_quaternion_xz(theta):
    """
    주어진 라디안(theta)을 xz 평면에서의 회전을 나타내는 쿼터니언 (x, y, z, w)으로 변환합니다.
    
    Parameters:
        theta (float): 회전 각도 (라디안 단위)
        
    Returns:
        tuple: (x, y, z, w) 형태의 쿼터니언
    """
    theta = theta[0]
    half_theta = theta / 2.0
    x = 0.0
    y = math.sin(half_theta)
    z = 0.0
    w = math.cos(half_theta)
    return [x, y, z, w]

def create_asset(name, class_name, category, position, rotation, scale):
    return {
        "name": name,
        "className": class_name,
        "category": category,
        "position": {"x": position[0], 
                     "y": position[1], 
                     "z": position[2]},
        "rotation": {"x": radian_to_quaternion_xz(rotation)[0], 
                     "y": radian_to_quaternion_xz(rotation)[1], 
                     "z": radian_to_quaternion_xz(rotation)[2], 
                     "w": radian_to_quaternion_xz(rotation)[3]},
        "scale": {"x": scale[0], 
                  "y": scale[1], 
                  "z": scale[2]},
    }

if __name__ == "__main__":
    ROOMS = ['bedroom', 'diningroom', 'livingroom']
    ROOMS = ['bedroom']

    for room_idx, room in enumerate(ROOMS):
        path = os.path.join(VALID_FOLDER_PATH, f'{room}/test/')
        pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
        
        for pkl_file in pkl_files[:3]:
            file_path = os.path.join(path, pkl_file)

            with open(file_path, 'rb') as file:
                pkl_data = pickle.load(file)

            scene_id = pkl_data['scene_id']
            layouts = pkl_data['layout']
            model_ids = pkl_data['model_ids']
            model_categorys = pkl_data['model_categorys']
            room_info = pkl_data['room_info']
            room_vertices = room_info['vertices']
            room_faces = room_info['faces']

            asset_list = []
            for layout, model_id, model_category in zip(layouts, model_ids, model_categorys):
                asset = create_asset(model_id, "StaticMeshActor", model_category, 
                                    layout[22:22+3],
                                    layout[22+3:22+4], 
                                    layout[22+4:22+7])
                asset_list.append(asset)

            data = {
                "assetList": asset_list,
                "roomVertices": room_vertices,
                "roomFaces": room_faces
            }
            # JSON 파일로 저장
            with open(f'{scene_id}.json', 'w') as json_file:
                data = json.loads(json.dumps(data, default=lambda x: float(x) if isinstance(x, np.float32) else x))
                json.dump(data, json_file, indent=4)

            print("JSON 파일이 성공적으로 생성되었습니다.")
