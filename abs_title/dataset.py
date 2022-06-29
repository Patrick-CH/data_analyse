import json

import torch
from torch.utils.data.dataset import Dataset

from utils import convert_to_id, pad_text


class AbsTitleDataSet(Dataset):
    def __init__(self, data_path='D:\\workspace\\PycharmProjects\\data_analyse\\data_abs\\train.json', empty=False,
                 val_split=0.2, device='cuda'):
        self.texts = []
        self.labels = []
        if not empty:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
                for _passage in data:
                    _title = _passage['title']
                    _content = _passage['content']
                    self.texts.append(_content)
                    self.labels.append(_title)

            self.texts = convert_to_id(self.texts)
            self.texts = pad_text(self.texts)
            self.labels = convert_to_id(self.labels)
            self.labels = pad_text(self.labels)

            self.texts = torch.tensor(self.texts, dtype=torch.long, device=device)
            self.labels = torch.tensor(self.labels, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        data = self.texts[index]
        label = self.labels[index]
        return data, label  # 返回数据和标签
