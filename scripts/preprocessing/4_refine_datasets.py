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

def generate_data_file(pkl_files, png_files, mode="train"):
    for pkl_file, png_file in zip(tqdm(pkl_files), png_files):
        data_path = os.path.join(path, pkl_file)
        image_condition = generate_resize_image(png_file)

        with open(data_path, 'rb') as file:
            pkl_data = pickle.load(file)
            scene_id = pkl_data['scene_id']
            room_info = pkl_data['room_info']
            object_infos = pkl_data['object_infos']
            object_transformations = pkl_data['object_transformations']
            velocity_field = pkl_data['velocity_field']

            text_condition = generate_text_condition(object_infos)
            refine_velocity_field = generate_refine_velocity_field(velocity_field)

            output_path = os.path.join(VALID_FOLDER_PATH, f"{mode}")
            os.makedirs(output_path, exist_ok=True)

            data = {
                "scene_id": scene_id,
                "text_condition": text_condition,
                "image_condition": image_condition,
                "gt_velocity_field": refine_velocity_field
            }
            output_file = os.path.join(VALID_FOLDER_PATH, f'{mode}/{scene_id}.pkl')
            with open(output_file, 'wb') as pkl_file:
                pickle.dump(data, pkl_file)

if __name__ == "__main__":
    path = os.path.join(VALID_FOLDER_PATH, f'velocity_field/')
    pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
    image_condition_files = [f for f in os.listdir(path) if f.endswith('condition.png')]

    train_pkl_files = pkl_files[:int(len(pkl_files) * 0.8)]  # 80%는 train
    train_png_files = image_condition_files[:int(len(image_condition_files) * 0.8)]  # 80%는 train
    val_pkl_files = pkl_files[int(len(pkl_files) * 0.8):]    # 20%는 val
    val_png_files = image_condition_files[int(len(image_condition_files) * 0.8):]    # 20%는 val

    generate_data_file(train_pkl_files, train_png_files, mode="train")
    generate_data_file(val_pkl_files, val_png_files, mode="val")
    
