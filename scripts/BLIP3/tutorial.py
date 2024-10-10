from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch

model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)

import json
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

# sample = {
#     "image_path": ["C:/Users/ttd85/Downloads/Southfield_grid_0_Google_Satellite.png"],
#     "question": ["Discribe this satelite aerial photograph <image>."]
# }

sample = {
    "image_path": ["E:/Resources/IndoorSceneSynthesis/InstructScene/3D-FRONT/3D-FUTURE-model/0a44f97c-ab24-44fc-a1e2-013fa1022115/image.jpg"],
    "question": ["Explain the types of furniture included in the image <image> and the purpose of each piece."]
}

# sample = {
#     "image_path": ["C:/Users/ttd85/Downloads/image1.jfif", "C:/Users/ttd85/Downloads/image2.jfif"],
#     "question": ["Discribe this satelite aerial photograph."]
# }

image_list = []
image_sizes = []
for fn in sample['image_path']:
    img = PIL.Image.open(fn)
    display.display(Image(filename=fn, width=300))
    
    image_tensor = image_processor([img], image_aspect_ratio='anyres')["pixel_values"].cuda().to(torch.bfloat16)
    image_list.append(image_tensor)
    image_sizes.append(img.size)

inputs = {
    "pixel_values": [image_list]
}
for query in sample['question']:
    prompt = apply_prompt_template(query)
    language_inputs = tokenizer([prompt], return_tensors="pt")
    inputs.update(language_inputs)
    # To cuda
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[name] = value.cuda()

    generated_text = model.generate(**inputs, image_size=[image_sizes],
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    temperature=0.05,
                                    do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1,
                                    )
    prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
    print("User: ", query)
    print("Assistant: ", textwrap.fill(prediction, width=100))
print("-"*120)