import re
from typing import List

import torch
from transformers import BertTokenizer, BertConfig, BertModel

MAX_SEN_LEN = 64

model_name = 'bert-base-chinese'

tokenizer = BertTokenizer.from_pretrained(model_name)
model_config = BertConfig.from_pretrained(model_name)

model_config.output_hidden_states = True
model_config.output_attentions = True

bert_model = BertModel(config=model_config)

bio2id = {'O': 0, 'B': 1, 'I': 2}
id2bio = ['O', 'B', 'I']


def label_id(labels):
    return [[bio2id[_j] for _j in _i] for _i in labels]


def split_text(text: str, max_len=MAX_SEN_LEN) -> List[str]:
    ls = re.split("[;,!，。；？！\n]", text)
    if max_len is not None:
        ls = [_i[:max_len] for _i in ls]
    return ls


def convert_to_id(texts: List[str]):
    text_tokens = [tokenizer.tokenize(_i) for _i in texts]
    text_ids = [tokenizer.convert_tokens_to_ids(_j) for _j in text_tokens]
    return text_ids


def pad_text(texts):
    return [i[:MAX_SEN_LEN] if len(i) >= MAX_SEN_LEN else (i + [tokenizer.pad_token_id] * (MAX_SEN_LEN - len(i))) for i in texts]


def pad_label(label):
    return [i[:MAX_SEN_LEN] if len(i) >= MAX_SEN_LEN else (i + ['O'] * (MAX_SEN_LEN - len(i))) for i in label]


def convert_to_vec(text_ids: List[int]):
    tokens_tensor = torch.tensor(text_ids)
    bert_model.eval()

    with torch.no_grad():
        outputs = bert_model(tokens_tensor)
        encoded_layers = outputs
        print(encoded_layers[0].shape, encoded_layers[1].shape)
