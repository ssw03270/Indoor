import os
import numpy as np
import pickle
import json
import random
from tqdm import tqdm

from utils import visualize_room_layout, generate_text_description, layout_to_code

folder_path = 'C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/InstructScene'
room_lists = ["threed_front_bedroom", "threed_front_diningroom", "threed_front_livingroom"]

random.seed(42)
np.random.seed(42)

for room in room_lists:
    overlap_count = 0
    error_count = 0    
    without_margin_room_data = []
    numerical_margin_room_data = []
    discrete_margin_room_data = []

    path = os.path.join(folder_path, room)
    
    # path 경로에 있는 폴더 리스트 가져오기
    folder_list = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and "test" not in f and "train" not in f]
    for folder in tqdm(folder_list):
        file_path = os.path.join(path, folder)
        boxes_npz = np.load(os.path.join(file_path, "boxes.npz"))
        with open(os.path.join(file_path, "descriptions.pkl"), "rb") as f:
            descriptions_pkl = pickle.load(f)
        with open(os.path.join(file_path, "models_info.pkl"), "rb") as f:
            models_info_pkl = pickle.load(f)
        relations_npy = np.load(os.path.join(file_path, "relations.npy"))

        # 데이터 파싱 및 구조화
        boxes_parsed_data = {
            'uids': boxes_npz['uids'].tolist(),                                 # shape: (modesl_count, 1) ex) ['82528/model', '82529/model', ...]
            'jids': boxes_npz['jids'].tolist(),                                 # shape: (modesl_count, 1) ex) ['7e79e5f8-d94b-4807-9676-4eb0e4e142aa', ...]
            'scene_id': boxes_npz['scene_id'].item(),                           # string, ex) 'DiningRoom-9450'
            'scene_uid': boxes_npz['scene_uid'].item(),                         # string, ex) 'ffb067ad-cf9a-4321-82ae-4e684c59ea3e_DiningRoom-9450' 
            'scene_type': boxes_npz['scene_type'].item(),                       # string, ex) 'diningroom'
            'json_path': boxes_npz['json_path'].item(),                         # string, ex) 'ffb067ad-cf9a-4321-82ae-4e684c59ea3e'
            'room_layout': boxes_npz['room_layout'].tolist(),                   # shape: (256, 256, 1) ex) [[[255.], [255.], [255.], ..., [255.]], ..., [[0.], [0.], [0.], ..., [255.]]]
            'floor_plan_vertices': boxes_npz['floor_plan_vertices'].tolist(),   # shape: (vertices_count, 3)
            'floor_plan_faces': boxes_npz['floor_plan_faces'].tolist(),         # shape: (faces_count, 3) ex) [[0, 1, 2], [0, 2, 3], ..., [1, 2, 3]]
            'floor_plan_centroid': boxes_npz['floor_plan_centroid'].tolist(),   # list, ex) [-2.225, 0.0, 2.83]
            'class_labels': boxes_npz['class_labels'].tolist(),                 # shape: (models_count, 26), one-hot encoding
            'translations': boxes_npz['translations'].tolist(),                 # shape: (models_count, 3) ex) [[0.8744308352470398, 0.7619104981422424, 0.13309000432491302], ...]
            'sizes': boxes_npz['sizes'].tolist(),                               # shape: (models_count, 3) ex) [[0.800000011920929, 0.38499993085861206, 0.5000029802322388], ...]
            'angles': boxes_npz['angles'].tolist()                              # shape: (models_count, 1) ex) [[0.0], [0.0], [1.5707871913909912]]
        }
        
        descriptions_parsed_data = {
            'obj_class_ids': descriptions_pkl['obj_class_ids'],                 # list, ex) [11, 10, 10, 10, 10, 7, 17, 23]
            'obj_counts': descriptions_pkl['obj_counts'],                       # list, ex) [(11, 1), (10, 4), (7, 1), (17, 1), (23, 1)]
            'obj_relations': descriptions_pkl['obj_relations']                  # list, ex) [(0, 4, 1), (0, 4, 2), (0, 9, 3), ...]
        }
        
        models_info_parsed_data = []
        for model in models_info_pkl:
            parsed_model = {
                'model_id': model['model_id'],                                  # string, ex) '7e79e5f8-d94b-4807-9676-4eb0e4e142aa'
                'super-category': model['super-category'],                      # string, ex) 'Table'
                'category': model['category'],                                  # string, ex) 'Dining Table'
                'style': model['style'],                                        # string, ex) 'Modern'
                'theme': model['theme'],                                        # string or None, ex) 'Texture Mark'
                'material': model['material'],                                  # string or None, ex) 'Solid Wood'
                'blip_caption': model['blip_caption'],                          # string, ex) 'a brown table with a wooden top'
                'msft_caption': model['msft_caption'],                          # string, ex) 'a wooden table with legs'
                'chatgpt_caption': model['chatgpt_caption'],                    # string, ex) 'a brown dining table with a wooden top'
                'objfeat_vq_indices': model['objfeat_vq_indices']               # list, ex) [26, 52, 1, 53]
            }
            models_info_parsed_data.append(parsed_model)

        relations_parsed_data = relations_npy.tolist()                          # shape: (relations_count, 3) ex) [[0, 4, 1], [0, 4, 2], [0, 9, 3], ...]
                
        # 영어 텍스트 설명 생성
        text_descriptions = generate_text_description(boxes_parsed_data['scene_type'], models_info_parsed_data)
        # print(f"생성된 영어 텍스트 설명: {text_description}")z

        # 방 레이아웃 시각화
        # visualize_room_layout(boxes_parsed_data, folder)
        try:
            without_margin_codes, numerical_margin_codes, discrete_margin_codes, is_overlap = layout_to_code(boxes_parsed_data, models_info_parsed_data, folder, debug=False, case_count=1)
            # print(f"생성된 코드:\n{generated_code}")

            if is_overlap:
                overlap_count += 1
                continue
            
            for text_description, without_margin_code, numerical_margin_code, discrete_margin_code in zip(
                text_descriptions, without_margin_codes, numerical_margin_codes, discrete_margin_codes):
                without_margin_room_data.append({
                    "scene_uid": boxes_parsed_data['scene_uid'],
                    "instruction": "Generate a room layout based on the given requirements.",
                    "input": text_description,
                    "output": without_margin_code
                })
                numerical_margin_room_data.append({
                    "scene_uid": boxes_parsed_data['scene_uid'],
                    "instruction": "Generate a room layout based on the given requirements.",
                    "input": text_description,
                    "output": numerical_margin_code
                })
                discrete_margin_room_data.append({
                    "scene_uid": boxes_parsed_data['scene_uid'],
                    "instruction": "Generate a room layout based on the given requirements.",
                    "input": text_description,
                    "output": discrete_margin_code
                })
                
        except:
            error_count += 1
            continue

    # overlap count와 error count 출력
    print(f"{room} 처리 결과:")
    print(f"  - 중복 발생 횟수: {overlap_count}")
    print(f"  - 오류 발생 횟수: {error_count}")
    print(f"  - 총 처리된 폴더 수: {len(folder_list)}")
    print(f"  - 성공적으로 처리된 폴더 수: {len(folder_list) - error_count - overlap_count}")
    print("-----------------------------")

    # 랜덤 인덱스 리스트 생성
    random_indices = list(range(len(without_margin_room_data)))
    random.shuffle(random_indices)

    without_margin_room_data = np.array(without_margin_room_data)[random_indices].tolist()
    numerical_margin_room_data = np.array(numerical_margin_room_data)[random_indices].tolist()
    discrete_margin_room_data = np.array(discrete_margin_room_data)[random_indices].tolist()

    # 각 방 유형별로 설명과 코드를 하나의 pkl 파일로 저장
    output_folder = os.path.join(folder_path, 'output')
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(path, f'{room}_without_margin_room_data.json')
    with open(output_file, 'w') as json_file:
        json.dump(without_margin_room_data, json_file, indent=4)
    print(f"{room}에 대한 설명과 코드가 {output_file}에 저장되었습니다.")

    output_file = os.path.join(path, f'{room}_numerical_margin_room_data.json')
    with open(output_file, 'w') as json_file:
        json.dump(numerical_margin_room_data, json_file, indent=4)
    print(f"{room}에 대한 설명과 코드가 {output_file}에 저장되었습니다.")
    
    output_file = os.path.join(path, f'{room}_discrete_margin_room_data.json')
    with open(output_file, 'w') as json_file:
        json.dump(discrete_margin_room_data, json_file, indent=4)
    print(f"{room}에 대한 설명과 코드가 {output_file}에 저장되었습니다.")