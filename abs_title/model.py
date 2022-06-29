import torch
from torch import nn
from transformers import BertModel, BertConfig


class KeyModel(nn.Module):
    def __init__(self, config):
        super(KeyModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        return sequence_output


class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):
        super().__init__()
        model_name = 'bert-base-chinese'
        self.bert = BertModel.from_pretrained(model_name)

        self.top_rnns = top_rnns
        self.fc = nn.Linear(768, vocab_size)

        self.device = device
        self.finetuning = finetuning
        # self.bert.to(self.device)

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64
        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)  # [128, 74]
        y = y.to(self.device)

        self.bert.eval()
        with torch.no_grad():
            encoded_layers = self.bert(x)[0]
            enc = encoded_layers[-1]  # [128, 74, 768]
        logits = self.fc(encoded_layers)  # [128, 74, 10]
        y_hat = logits.argmax(-1)  # [128, 74]
        # return logits, y, y_hat
        return logits
