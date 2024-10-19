import os
import pickle
import numpy as np
from tqdm import tqdm

from PIL import Image

from shapely.geometry import Polygon, Point, LineString
from scipy.interpolate import griddata
from scipy.ndimage import convolve, binary_erosion, gaussian_filter

import matplotlib.pyplot as plt

from utils import a_star_with_radius, dijkstra_with_radius

np.random.seed(42)

# Define global paths and room types
VALID_FOLDER_PATH = "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes"
bedroom_class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe", "none"]
diningroom_class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "none"]
livingroom_class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "none"]

class_label_list = [bedroom_class_labels, diningroom_class_labels, livingroom_class_labels]
max_obj_count_list = [12, 21, 21]

def generate_text_condition(object_infos):
    text_condition = ""

    caption_dict = {}
    for object_info in object_infos:
        caption = object_info['chatgpt_caption']

        if caption not in caption_dict:
            caption_dict[caption] = 1
        else:
            caption_dict[caption] += 1
    
    # 영어로 text_condition 생성
    for caption, count in caption_dict.items():
        if count > 1:
            if caption[0] == 'a' and caption[1] == ' ':
                text_condition += f"{count} {caption[2:]}(s), "
            else:
                text_condition += f"{count} {caption}(s), "
        else:
            text_condition += f"{caption}, "
    
    text_condition = text_condition.rstrip(", ")  # 마지막 쉼표와 공백 제거
    text_condition += ". "

    return text_condition  # text_condition 반환

def generate_refine_velocity_field(velocity_field):
    U_smooth = velocity_field['U_smooth']
    V_smooth = velocity_field['V_smooth']
    
    # Compute the padding values, making sure they are integers
    pad_x1 = (64 - U_smooth.shape[0]) // 2
    pad_x2 = 64 - U_smooth.shape[0] - pad_x1
    pad_y1 = (64 - U_smooth.shape[1]) // 2
    pad_y2 = 64 - U_smooth.shape[1] - pad_y1

    # Pad the array using integer padding values
    U_smooth = np.pad(U_smooth, ((pad_x1, pad_x2), (pad_y1, pad_y2)), mode='constant')
    V_smooth = np.pad(V_smooth, ((pad_x1, pad_x2), (pad_y1, pad_y2)), mode='constant')

    return np.array([U_smooth, V_smooth])

def generate_resize_image(png_file):
    image = plt.imread(os.path.join(path, png_file))  # PNG 파일을 로드
    image = image[:, :, :3]
    image = Image.fromarray((image * 255).astype(np.uint8))
    image = image.resize((64, 64), Image.NEAREST)
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)

    return image

def generate_data_file(room, pkl_files, png_files, class_labels, max_obj_count, mode="train"):
    one_hot_dict = {label: [1 if i == idx else 0 for i in range(len(class_labels))] 
                 for idx, label in enumerate(class_labels)}

    for pkl_file, png_file in zip(tqdm(pkl_files), png_files):
        data_path = os.path.join(path, pkl_file)
        image_condition = generate_resize_image(png_file)

        with open(data_path, 'rb') as file:
            pkl_data = pickle.load(file)
            
        split = pkl_data['split']
        if split != mode:
            continue

        scene_id = pkl_data['scene_id']
        room_info = pkl_data['room_info']
        object_infos = pkl_data['object_infos']
        object_transformations = pkl_data['object_transformations']
        velocity_field = pkl_data['velocity_field']

        model_ids = [object_info['model_id'] for object_info in object_infos]
        categorys = [object_transformation['category'] for object_transformation in object_transformations]
        locations = [object_transformation['location'] for object_transformation in object_transformations]
        scales = [object_transformation['scale'] for object_transformation in object_transformations]
        rotations = [object_transformation['rotation'] for object_transformation in object_transformations]

        obj_outputs = []
        for cat, loc, scale, rot in zip(categorys, locations, scales, rotations):
            cat_one_hot = one_hot_dict[cat]
            obj_output = cat_one_hot + loc + scale + [rot]
            obj_outputs.append(obj_output)
        
        if len(obj_outputs) < max_obj_count:
            obj_outputs.append(one_hot_dict['none'] + [0, 0, 0, 0, 0, 0, 0])

        text_condition = generate_text_condition(object_infos)
        refine_velocity_field = generate_refine_velocity_field(velocity_field)

        output_path = os.path.join(VALID_FOLDER_PATH, f"{room}/{mode}")
        os.makedirs(output_path, exist_ok=True)

        data = {
            "scene_id": scene_id,
            "room_info": room_info,
            "text_condition": text_condition,
            "image_condition": image_condition,
            "gt_velocity_field": refine_velocity_field,
            "layout": obj_outputs,
            "model_ids": model_ids,
            "model_categorys": categorys
        }
        output_file = os.path.join(VALID_FOLDER_PATH, f'{room}/{mode}/{scene_id}.pkl')
        with open(output_file, 'wb') as pkl_file:
            pickle.dump(data, pkl_file)

if __name__ == "__main__":
    ROOMS = ['bedroom', 'diningroom', 'livingroom']

    for room_idx, room in enumerate(ROOMS):
        path = os.path.join(VALID_FOLDER_PATH, f'{room}_velocity_field/')
        pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
        image_condition_files = [f for f in os.listdir(path) if f.endswith('condition.png')]
        class_labels = class_label_list[room_idx]
        max_obj_count = max_obj_count_list[room_idx]

        generate_data_file(room, pkl_files, image_condition_files, class_labels, max_obj_count, mode="train")
        generate_data_file(room, pkl_files, image_condition_files, class_labels, max_obj_count, mode="test")
        generate_data_file(room, pkl_files, image_condition_files, class_labels, max_obj_count, mode="val")
        
