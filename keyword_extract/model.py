from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertModel


class KeyModel(nn.Module):
    def __init__(self, config, n_tag):
        super(KeyModel, self).__init__()
        self.bert = BertModel(config)
        self.bert.train(False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.d0 = nn.Linear(config.hidden_size, 256)
        self.d1 = nn.Linear(256, 16)
        self.d2 = nn.Linear(16, n_tag)
        self.num_labels = n_tag

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.d0(sequence_output)
        sequence_output = self.d1(sequence_output)
        logits = self.d2(sequence_output)
        return logits
