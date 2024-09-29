# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import json

from tqdm import tqdm

import torch.nn as nn
import torch
from transformers import LlamaTokenizer

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaConfig,
)

from llama_recipes.inference.chat_utils import read_dialogs_from_file
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker

B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_dialog(dialog, tokenizer):
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog]

    # dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    dialog_tokens = [dialog_token for dialog_token in prompt_tokens]

    return dialog_tokens

def load_json_dataset(json_file, split="validation"):
    with open(json_file, 'r') as f:
        data = json.load(f)

    dialog_data = []
    gt_data = []
    for entry in data:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        output_text = entry.get("output", "")

        # 사용자와 어시스턴트 역할 설정
        dialog_data.append({"role": "user", "content": f"{B_INST} {instruction} {E_INST} {input_text}"})
        gt_data.append({"role": "assistant", "content": output_text})

    return dialog_data, gt_data

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=False, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    batch_size: int = 4,  # 배치 크기 추가
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs, gts= load_json_dataset(prompt_file)

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    # print(f"User dialogs:\n{dialogs}")
    print("\n==================================\n")


    # GPU 사용 가능 여부 확인 및 사용 가능한 GPU 수 파악
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU 수: {num_gpus}")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization, use_fast_kernels=False)
    if peft_model:
        model = load_peft_model(model, peft_model)
    
    # 모델을 여러 GPU에 분산
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    
    chats = tokenize_dialog(dialogs, tokenizer)
    outputs_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(chats), batch_size), desc="배치 처리 중"):
            batch_chats = chats[i:i+batch_size]
            
            # 배치 내 최대 길이로 패딩
            max_length = max(len(chat) for chat in batch_chats)
            padded_chats = [chat + [tokenizer.pad_token_id] * (max_length - len(chat)) for chat in batch_chats]
            
            tokens = torch.tensor(padded_chats).long().to(device)
            
            # DataParallel을 사용할 경우 model.module을 사용
            generate_func = model.module.generate if isinstance(model, nn.DataParallel) else model.generate
            
            # 배치 단위로 생성
            outputs = generate_func(
                input_ids=tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

            # 배치 출력 처리
            for output in outputs:
                output_text = tokenizer.decode(output, skip_special_tokens=True)
                outputs_list.append(output_text)
                print(f"Model output:\n{output_text}")
                print("\n==================================\n")
                
    result = []
    for dialog, output in zip(dialogs, outputs_list):
        result.append({
            "input": dialog["content"],
            "output": output
        })
    
    with open('inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("결과가 'inference_results.json' 파일에 저장되었습니다.")



if __name__ == "__main__":
    fire.Fire(main)