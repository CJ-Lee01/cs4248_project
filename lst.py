import pandas as pd
import sys
import time
from typing import List
import numpy as np
import scipy.sparse
import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
print(device)

if __name__ == '__main__':
    pass

train_fpath = './data/xtrain.txt'
test_fpath = './data/balancedtest.csv'

train_raw_data = pd.read_csv(train_fpath, names=('label', 'text'), delimiter='\t')
test_raw_data = pd.read_csv(test_fpath, names=('label', 'text'))

from string import punctuation

def transform_str_data(text: pd.Series):
    t: pd.Series = text.str.lower()
    t: pd.Series = t.str.replace('\n', ' ')
    t: pd.Series = t.str.replace(r'[!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~-]', ' ', regex=True)
    return t

train_raw_data['text'] = transform_str_data(train_raw_data['text'])
test_raw_data['text'] = transform_str_data(test_raw_data['text'])

from torch.utils.data import DataLoader, Dataset


class RawTextDataset(Dataset):

    def __init__(self, data: pd.DataFrame):
        sentences = list(data['text'].str.split())
        labels = list(data['label'].transform(lambda x: x - 1))
        self.data = list(zip(sentences, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

training_dataset = RawTextDataset(train_raw_data)
test_dataset = RawTextDataset(test_raw_data)


def yield_tokens(dataset: RawTextDataset) -> List[str]:
    for i in range(len(dataset)):
        sentence, _ = dataset[i]
        yield sentence

vocab = build_vocab_from_iterator(yield_tokens(training_dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: vocab(x)

seq_length = 1000

def pad_sequence(text):
    seq = np.zeros(seq_length, dtype=int)
    seq[-len(text):] = np.array(text_pipeline(text))[0:seq_length]
    return seq

def collate_batch(batch):
    label_list, text_list = [], []
    for text, label in batch:
        label_list.append(label)
        processed_text = torch.tensor(pad_sequence(text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # Pads inputs vectors to be of same length (seq_length)
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    # Label list size: [batch_size]
    # Text list size: [batch_size, seq_length]
    return label_list.to(device), text_list.to(device)


class BasicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout_prob = 0.3, bidirectional=True):
        super(BasicRNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                           dropout=dropout_prob, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.rest = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.Dropout(dropout_prob),
            nn.Linear(100, 4)
        )
        self.fc = nn.Linear(hidden_dim, 4)

    def init_hidden(self, batch_size):
        # Note that batch_size must be passed in from the training loop
        # to handle the case where the last batch has fewer data.
        # hidden state and cell state size:
        # [num_layers, batch_size, hidden_size]
        weight = next(self.parameters()).data

        dimensions = 2 if self.bidirectional else 1

        return weight.new(dimensions * self.n_layers, batch_size, self.hidden_dim).zero_().to(device)


    def forward(self, text, hidden):
        # Embedding vector size: [batch_size, seq_length, embed_dim]
        embeds = self.embedding(text)
        # LSTM output size: [batch_size, seq_length, (2 if bidirectional else 1) * hidden_size]
        # h_n & c_n output size: [(2 if bidirectional else 1) * num_layers, batch_size, hidden_size]
        # h_n output size: [1, batch_size, hidden_size]
        rnn_out, hidden = self.rnn(embeds, hidden)

        # Get last batch of labels
        # Hidden state size: [batch_size, hidden_size]
        hidden_state = hidden[0]
        hidden_state = self.dropout(hidden_state)

        # Linear layer output size: [batch_size, 4]
        output = self.rest(hidden_state)

        return output, hidden

# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 100
n_layers = 2
dropout_prob = 0.8
bidirectional = False
EPOCHS = 10 # epoch
LR = 0.001  # learning rate
BATCH_SIZE = 64 # batch size for training

model = BasicRNN(vocab_size, embedding_dim, hidden_dim, n_layers, dropout_prob=dropout_prob, bidirectional=bidirectional)
model.to(device)

print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)


def train(dataloader):
    # initialize hidden state
    h = model.init_hidden(BATCH_SIZE)

    model.train()
    total_acc, total_count, train_loss = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = h.data

        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        output, h = model(text, h)

        # calculate the loss and perform backprop
        loss = criterion(output, label)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*text.size(0)
        # update training accuracy
        total_acc += (output.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | loss {:8.3f}'
                  '| accuracy {:8.3f}'.format(epoch_num, idx, length(dataloader),
                                              train_loss, total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    # initialize hidden state
    val_h = model.init_hidden(BATCH_SIZE)
    model.eval()
    total_acc, total_count = 0, 0
    tp, fp, fn = [0] * 4, [0] * 4, [0] * 4
    results = []

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            val_h = val_h.data
            output, val_h = model(text, val_h)

            predicted_label = output.argmax(1)
            total_acc += (predicted_label == label).sum().item()
            total_count += label.size(0)

            temp = list(zip(list(predicted_label), list(label)))
            results.extend(list(zip(text, temp)))

            for lab in range(4):
                tp[lab] += torch.mul(predicted_label == lab, label == lab).sum().item()
                fp[lab] += torch.mul(predicted_label == lab, label != lab).sum().item()
                fn[lab] += torch.mul(predicted_label != lab, label == lab).sum().item()

    f1s = [tp[lab] / (tp[lab] + 0.5 * (fp[lab] + fn[lab])) for lab in range(4)]
    return total_acc/total_count, f1s[0], f1s[1], f1s[2], f1s[3], sum(f1s) / 4, results

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
epoch_num = 0
num_train = int(len(training_dataset) * 0.95)
split_train_: Subset[RawTextDataset]
split_valid_: Subset[RawTextDataset]
split_train_, split_valid_ = random_split(training_dataset, [num_train, len(training_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch, drop_last=True)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch, drop_last=True)

def run_one_epoch():
    # Need to explicitly define these variables as global so that Python does not create local variables when updating.
    global epoch_num, total_accu

    epoch_num += 1
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val, f1_0, f1_1, f1_2, f1_3, f1_macro, results = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        print("Learning rate reduced")
        scheduler.step()
    else:
        total_accu = accu_val

    print('-' * 181)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} '
          '| f1 of class 0 {:8.3f} | f1 of class 1 {:8.3f} | f1 of class 2 {:8.3f} '
          '| f1 of class 3 {:8.3f} | macro f1 {:8.3f} |'.format(epoch_num,
                                              time.time() - epoch_start_time,
                                              accu_val, f1_0, f1_1, f1_2, f1_3, f1_macro))
    print('-' * 181)


for epoch in range(1, EPOCHS + 1):
    run_one_epoch()

print('Checking the results of test dataset.')
accu_test, f1_0_test, f1_1_test, f1_2_test, f1_3_test, f1_macro_test, results = evaluate(test_dataloader)
print('test accuracy {:8.3f} | test f1 of class 0 {:8.3f} | test f1 of class 1 {:8.3f} '
      '| test f1 of class 2 {:8.3f} | test f1 of class 3 {:8.3f} | test macro f1 {:8.3f} |'.format(accu_test,
      f1_0_test, f1_1_test, f1_2_test, f1_3_test, f1_macro_test))


# Confusion matrix
confusion_matrix = [[0, 0, 0, 0] for i in range(4)]
for text, (predicted, actual) in results:
    confusion_matrix[actual][predicted] += 1

import seaborn as sn
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(confusion_matrix, range(4), range(4))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

# plt.show()

correct = []
incorrect = []
acc_dict = {}


for text, (predicted, actual) in results:
    length = ((torch.count_nonzero(text).cpu() // 50) * 50).item()
    if length not in acc_dict:
        acc_dict[length] = [0, 0]
    if predicted == actual:
        acc_dict[length][0] += 1
    else:
        acc_dict[length][1] += 1

    if predicted == actual:
        correct.append(torch.count_nonzero(text).cpu() // 50)
    else:
        incorrect.append(torch.count_nonzero(text).cpu() // 50)

x_axis = []
y_axis = []
for key in sorted(acc_dict.keys()):
    x_axis.append(key)
    y_axis.append(acc_dict[key][0] / (sum(acc_dict[key]) + 1))


from matplotlib import pyplot
pyplot.plot(x_axis, y_axis)
# pyplot.show()


params = model.embedding.parameters()
p = (list(params)[0]).cpu().detach().numpy()
p = p / np.linalg.norm(p, axis=1)[:,np.newaxis]

for i in range(len(vocab)):
    token = vocab.lookup_token(i)
    lst = p[i] @ p.T
    similiars = [v for v in vocab.lookup_tokens(list(np.argpartition(lst, -5)[-5:]))]
    print(token, similiars)
