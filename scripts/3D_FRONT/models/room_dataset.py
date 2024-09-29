import copy
import datasets
import itertools
import json

B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_dialog(dialog, tokenizer):
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
    answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
    
    # dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    dialog_tokens = [dialog_token + answer_token for dialog_token, answer_token in zip(prompt_tokens, answer_tokens)]
    labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i, c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": dialog_tokens,
        "labels": labels_tokens,
    }

    # attention_mask도 이중 리스트로 변경
    attention_mask = [[1] * len(t) for t in dialog_tokens]

    return dict(combined_tokens, attention_mask=attention_mask)

def load_json_dataset(json_file, split):
    with open(json_file, 'r') as f:
        data = json.load(f)

    eval_length = int(len(data)/20)
    if split == "train":
        return data[:-eval_length]
    else:
        return data[-eval_length:]

def get_custom_dataset(dataset_config, tokenizer, split):
    # JSON 파일 로드
    data = load_json_dataset(dataset_config.data_path, split)

    # 대화형 데이터로 변환
    dialog_data = []
    for entry in data:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        output_text = entry.get("output", "")

        # 사용자와 어시스턴트 역할 설정
        dialog_data.append({"role": "user", "content": f"{B_INST} {instruction} {E_INST} {input_text}"})
        dialog_data.append({"role": "assistant", "content": output_text})

    # 토큰화
    dataset = tokenize_dialog(dialog_data, tokenizer)
    # print("tokenize_dialog 결과 미리보기:")
    # print("input_ids (처음 10개):", dataset['input_ids'][:2])
    # print("labels (처음 10개):", dataset['labels'][:2])
    # print("attention_mask (처음 10개):", dataset['attention_mask'][:2])
    
    # 반환할 데이터셋 구성
    return datasets.Dataset.from_dict(dataset)