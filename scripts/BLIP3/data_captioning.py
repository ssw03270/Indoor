import os
import json
from tqdm import tqdm

from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch

model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)

import PIL
import textwrap
import IPython.display as display
from IPython.display import Image

def apply_prompt_template(prompt):
    s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
            )
    return s 

model = model.to('cuda')
model.eval()
tokenizer.padding_side = "left"
tokenizer.eos_token = '<|end|>'

# query = """
# "Can you provide a detailed description of this piece of furniture <image> ? Please include the following aspects in your explanation:
# 1. Shape 2. Color 3. Material 4. Purpose of use 5.Presence of storage space

# Additionally, here are a example of how to answer based on different types of furniture:"

# Shape: This is a three-seater sofa with a rectangular structure. It has large, rounded arms and a low back.
# Color: The sofa is a deep navy blue.
# Material: The sofa is made of soft fabric with a suede-like texture. The legs are wooden and stained dark brown.
# Purpose of use: It is designed for seating in a living room or lounge area. It provides comfort for multiple people.
# Presence of storage space: There is no built-in storage in this sofa.
# """
query = """
I'm trying to arrange the [caption content] drawn in the image <image> in the room. 
Select all directions from which a person can approach this furniture to use it: (forward, right, backward, left).
"""

dataset_path = "E:/Resources/IndoorSceneSynthesis/LayoutGPT/Preprocessed_bedroom/discrete_margin"
dataset_name = "bedroom_train_data.json"

image_dataset_path = "E:/Resources/IndoorSceneSynthesis/3D-FUTURE-model"
image_dataset_name = "image.jpg"

dataset_file = os.path.join(dataset_path, dataset_name)

# JSON 파일 읽기
with open(dataset_file, 'r') as f:
    dataset = json.load(f)

object_captions = {}

# output 폴더 경로 설정
output_folder = "E:/Resources/IndoorSceneSynthesis/ObjectCaption"
os.makedirs(output_folder, exist_ok=True)

for data in tqdm(dataset):
    object_ids = data['object_ids']
    gpt_captions = data['object_captions']

    for object_id, gpt_caption in zip(object_ids, gpt_captions):
        if object_id not in object_captions:
            image_file_path = os.path.join(image_dataset_path, f"{object_id}", image_dataset_name)
            try:
                image = PIL.Image.open(image_file_path)
            except FileNotFoundError:
                print(f"경고: 이미지 파일을 찾을 수 없습니다: {image_file_path}")
                continue
            
            image_tensor = [image_processor([image], image_aspect_ratio='anyres')["pixel_values"].cuda().to(torch.bfloat16)]
            image_size = [image.size]

            inputs = {
                "pixel_values": [image_tensor]
            }

            current_query = query.replace("caption content", f"{gpt_caption}")
            prompt = apply_prompt_template(current_query)
            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)
            # To cuda
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[name] = value.cuda()

            generated_text = model.generate(**inputs, image_size=[image_size],
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            temperature=0.05,
                                            do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1,
                                            )
            prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]

            print("Object ID: ", object_id)
            print("User: ", current_query)
            print("Assistant: ", textwrap.fill(prediction, width=100))
            print()

            object_captions[object_id] = prediction

            # txt 파일로 저장
            output_file = os.path.join(output_folder, f"{object_id}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(prediction)

        else:
            caption = object_captions[object_id]