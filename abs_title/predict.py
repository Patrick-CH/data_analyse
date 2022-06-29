import torch
from transformers import BertConfig, BertTokenizer

from model import KeyModel
from utils import convert_to_id, pad_text

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('正在使用显卡')
else:
    device = torch.device('cpu')
    print('正在使用cpu')


if __name__ == '__main__':
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model_config = BertConfig.from_pretrained(model_name)

    model_config.output_hidden_states = True
    model_config.output_attentions = True

    model = KeyModel(config=model_config, n_tag=3)
    model.load_state_dict(torch.load('models/model1'))
    model.to(device)

    texts_plain = ['在这样的背景下', '中国倡导的共商共建共享的全球治理观', '可谓破解“治理赤字”的一剂良方。',
                   '让各族群众都过上好日', '必须付出艰苦努', '离不开各级党组织的坚强领导', '离不开各族群众的团结奋斗']
    texts = convert_to_id(texts_plain)
    texts = pad_text(texts)

    texts = torch.tensor(texts, dtype=torch.long, device=device)

    output = model(texts)
    res = torch.argmax(output, dim=-1)
    for idx, it in enumerate(res):
        print(texts_plain[idx], it)
