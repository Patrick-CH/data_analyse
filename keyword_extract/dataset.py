import json

import jieba
import torch
from torch.utils.data.dataset import Dataset

from utils import split_text, convert_to_id, convert_to_vec, pad_text, bio2id, pad_label, label_id


class KeyWordDataSet(Dataset):
    def __init__(self, data_path='D:\\workspace\\PycharmProjects\\data_analyse\\data\\train.json', empty=False,
                 val_split=0.2, device='cuda'):
        self.texts = []
        self.labels = []
        if not empty:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
                for _passage in data:
                    _title = _passage['title']
                    _content = _passage['content']
                    texts_i = split_text(_content)
                    texts_i_ = []
                    for ti in texts_i:
                        if len(ti) != 0:
                            texts_i_.append(ti)
                    texts_i = texts_i_
                    labels_i = []
                    keywords = [_i for _i in jieba.cut(_title)]
                    for t in texts_i:
                        label = ['O' for i in t]
                        for kwo in keywords:
                            if kwo in t:
                                b_idx = t.index(kwo)
                                label[b_idx] = 'B'
                                for _idx in range(1, len(kwo)):
                                    label[b_idx + _idx] = 'I'
                        labels_i.append(label)
                    self.texts.extend(texts_i)
                    self.labels.extend(labels_i)

            self.texts = convert_to_id(self.texts)
            self.texts = pad_text(self.texts)

            self.labels = pad_label(self.labels)
            self.labels = label_id(self.labels)

            self.texts = torch.tensor(self.texts, dtype=torch.long, device=device)
            self.labels = torch.tensor(self.labels, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        data = self.texts[index]
        label = self.labels[index]
        return data, label  # 返回数据和标签
