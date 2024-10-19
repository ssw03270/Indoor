import os
import numpy as np
import pickle
import json
import random
from tqdm import tqdm
import csv

from utils import check_overlap

folder_path = 'E:/Resources/IndoorSceneSynthesis/InstructScene'
room_lists = ["threed_front_bedroom", "threed_front_diningroom", "threed_front_livingroom"]
split_paths = ["E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/bedroom_threed_front_splits.csv",
               "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/diningroom_threed_front_splits.csv",
               "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/livingroom_threed_front_splits.csv"]
random.seed(42)
np.random.seed(42)

for room, split_path in zip(room_lists, split_paths):
    overlap_count = 0
    out_of_bound_count = 0
    error_count = 0    
    valid_scene_infos = []

    obj_count_list = []

    path = os.path.join(folder_path, room)
    with open(split_path, mode='r', encoding='utf-8') as csvfile:  # CSV 파일 열기
        reader = csv.reader(csvfile)
        split_data = list(reader)
        split_data = {item[0]: item[1] for item in split_data}

    # path 경로에 있는 폴더 리스트 가져오기
    folder_list = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and "test" not in f and "train" not in f]
    all_folder_count = len(folder_list)
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
                'category': model['category'],                                  # string, ex) 'Dining Table'
                'chatgpt_caption': model['chatgpt_caption'],                    # string, ex) 'a brown dining table with a wooden top'
            }
            models_info_parsed_data.append(parsed_model)

        relations_parsed_data = relations_npy.tolist()                          # shape: (relations_count, 3) ex) [[0, 4, 1], [0, 4, 2], [0, 9, 3], ...]
                
        # 방 레이아웃 시각화
        # visualize_room_layout(boxes_parsed_data, folder)
        # try:
        #     is_overlap, is_out_of_bound = check_overlap(boxes_parsed_data, models_info_parsed_data, folder, debug=False, case_count=1)
        #     if is_overlap:
        #         overlap_count += 1
            
        #     if is_out_of_bound:
        #         out_of_bound_count += 1
            
        #     if is_overlap or is_out_of_bound:
        #         all_folder_count -= 1
        #         continue
            
        #     valid_scene_infos.append({
        #         'scene_id': boxes_parsed_data['scene_uid'],
        #         'object_infos': models_info_parsed_data,
        #     })
                
        # except:
        #     error_count += 1
        #     all_folder_count -= 1
        #     continue

        valid_scene_infos.append({
            'split': split_data[boxes_parsed_data['scene_id']],
            'scene_id': boxes_parsed_data['scene_uid'],
            'object_infos': models_info_parsed_data,
        })

        obj_count_list.append(len(models_info_parsed_data))

    # overlap count와 error count 출력
    print(f"{room} 처리 결과:")
    print(f"  - 중복 발생 횟수: {overlap_count}")
    print(f"  - 넘침 발생 횟수: {out_of_bound_count}")
    print(f"  - 오류 발생 횟수: {error_count}")
    print(f"  - 총 처리된 폴더 수: {len(folder_list)}")
    print(f"  - 성공적으로 처리된 폴더 수: {all_folder_count}")

        # 최대, 최소, 평균 출력
    max_count = max(obj_count_list)  # 최대값
    min_count = min(obj_count_list)  # 최소값
    avg_count = sum(obj_count_list) / len(obj_count_list) if obj_count_list else 0  # 평균값

    print(f"{room}의 객체 수 통계:")
    print(f"  - 최대 객체 수: {max_count}")
    print(f"  - 최소 객체 수: {min_count}")
    print(f"  - 평균 객체 수: {avg_count:.2f}")
    
    print("-----------------------------")

    # 각 방 유형별로 설명과 코드를 하나의 pkl 파일로 저장
    output_folder = os.path.join(folder_path, 'valid_scenes')
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, f'{room}_valid_scenes.json')
    with open(output_file, 'w') as json_file:
        json.dump(valid_scene_infos, json_file, indent=4)
    print(f"{room}에 대한 정보가 {output_file}에 저장되었습니다.")