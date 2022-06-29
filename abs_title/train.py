from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import AbsTitleDataSet
from transformers import BertTokenizer, BertConfig, BertModel

from abs_title.model import KeyModel, Net

MAX_SEN_LEN = 64
BATCH = 16
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('正在使用显卡')
else:
    device = torch.device('cpu')
    print('正在使用cpu')


def train_one_epoch(epoch_index, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        outputs = outputs.reshape(-1, outputs.shape[-1])  # (N*T, VOCAB)
        labels = labels.view(-1)  # (N*T,)
        loss = loss_fn(outputs, labels)
        if i == 0:
            print(outputs.size())
            print(labels.size())

        loss.backward()
        running_loss += loss.item()
        optimizer.step()

        if i % 100 == 99:
            torch.cuda.empty_cache()
            last_loss = running_loss / 100  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


if __name__ == '__main__':
    train_data = AbsTitleDataSet(device=device)
    val_data = AbsTitleDataSet(data_path='D:\workspace\PycharmProjects\data_analyse\data_abs\dev.json', device=device)
    print(len(val_data), len(train_data))

    training_loader = DataLoader(train_data, batch_size=BATCH)
    validation_loader = DataLoader(val_data, batch_size=BATCH)

    model = Net(device=device, vocab_size=BertConfig.from_pretrained('bert-base-chinese').vocab_size)
    model.to(device=device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, optimizer, loss_fn)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs, vlabels)

            voutputs = voutputs.reshape(-1, voutputs.shape[-1])  # (N*T, VOCAB)
            vlabels = vlabels.view(-1)  # (N*T,)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / len(val_data)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
