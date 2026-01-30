from os import PathLike
from typing import List

import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    pass
print(torch.cuda.is_available())
CTX = torch.device('cuda')


TRAIN_PATH: PathLike[str] = '../data/xtrain.txt'
TEST_PATH: PathLike[str] = '../data/balancedtest.csv'
BATCH_SIZE = 1

tokenizer = get_tokenizer('basic_english')

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

def yield_tokens(str_lst: List[str]):
    # https://pytorch.org/text/stable/vocab.html?highlight=build_vocab#torchtext.vocab.build_vocab_from_iterator
    for line in str_lst:
        yield tokenizer(line)



class TextDataset(Dataset):
    def __init__(self, file: PathLike[str], *, sep=',', encoder=None, scramble=True):
        self.data = pd.read_csv(file, sep=sep)
        if scramble:
            self.data = self.data.sample(frac=1)
        self.encoder = build_vocab_from_iterator(yield_tokens(list(self.data.iloc[:, 1])), specials=['<unk>']) if encoder is None else encoder
        if encoder is None:
            self.encoder.set_default_index(self.encoder['<unk>'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        target = self.data.iloc[item, 0]
        text = self.data.iloc[item, 1]
        return target - 1, (self.encoder(tokenizer(text))[::-1], len(tokenizer(text)))

RAW = TextDataset(TRAIN_PATH, sep='\t')
ENCODER = RAW.encoder
train_data = DataLoader(RAW, batch_size=BATCH_SIZE)

class LSTMClassifier(nn.Module):

    def __init__(self, *, vocab_size, embedding_dim=4, hidden_dim=4, output_dim=4, num_layers=1, dropout=0.2, bidirectional=False):
        super(LSTMClassifier, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.rest = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, sequence):
        h = torch.zeros((self.num_layers, sequence.size(0), self.hidden_dim))
        c = torch.zeros((self.num_layers, sequence.size(0), self.hidden_dim))

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        embedded = self.embedding(sequence)

        packed_output, (hidden_state, cell_state) = self.lstm(embedded)
        result = self.rest(packed_output)
        return result

import argparse

parser = argparse.ArgumentParser(prog='CS4248 LSTM')
parser.add_argument('-EMB', '--EMBEDDING_SIZE', default=64, type=int)
parser.add_argument('-HID', '--HIDDEN', default=16, type=int)
parser.add_argument('-L', '--NUM_LAYERS', default=1, type=int)
parser.add_argument('-BI', '--BIDIRECTIONAL', default=False, action='store_true')
parser.add_argument('-n', '--name', required=True)

args = vars(parser.parse_args())
print(args)
VOCAB_SIZE = len(RAW.encoder)
EMBEDDING_DIM = args['EMBEDDING_SIZE']
NUM_HIDDEN_NODES = args['HIDDEN']
NUM_OUTPUT_NODES = 4
NUM_LAYERS = args['NUM_LAYERS']
DROPOUT = 0.5
BIDIRECTIONAL = args['BIDIRECTIONAL']

model = LSTMClassifier(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=NUM_HIDDEN_NODES,
                       output_dim=NUM_OUTPUT_NODES, num_layers=NUM_LAYERS, dropout=DROPOUT,
                       bidirectional=BIDIRECTIONAL)

import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, *, limit=None, step=1000):
    epoch_loss = 0.0
    epoch_acc = 0.0
    loss = 0
    ctr = 0

    model.train()


    for batch in iterator:
        ctr += 1

        if step is not None and ctr % step == 0:
            print(ctr, loss)
        if ctr is not None and ctr == limit:
            break
        # cleaning the cache of optimizer
        optimizer.zero_grad()

        type, (text, text_lengths) = batch
        # forward propagation and squeezing
        predictions = model(text, text_lengths)

        # computing loss / backward propagation
        loss = criterion(predictions, type)
        loss.backward()

        # accuracy
        acc = binary_accuracy(predictions, type)

        # updating params
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # It'll return the means of loss and accuracy
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# We'll use this helper to compute accuracy
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0.0
    epoch_acc = 0.0

    # deactivate the dropouts
    model.eval()

    labels, preds = [], []

    # Sets require_grad flat False
    with torch.no_grad():
        for batch in iterator:
            label, (text, text_lengths) = batch
            labels.append(label.item())
            prediction = model(text, text_lengths).squeeze().argmax().item()
            preds.append(prediction)
            # compute loss and accuracy

    return [round(n, 4) for n in sklearn.metrics.f1_score(preds, labels, average=None)], preds


RAW_TEST = TextDataset(TEST_PATH, encoder=ENCODER, scramble=False)
validation_iterator = DataLoader(RAW_TEST, batch_size=BATCH_SIZE)
#report = open('REPORT', 'a')
#report.write(f'\nLSTM {args["name"]}\n{args}\n')

EPOCH_NUMBER = 10
for epoch in range(1, EPOCH_NUMBER + 1):
    train_loss, train_acc = train(model, train_data, optimizer, criterion, limit=500)

    valid_f1, preds = evaluate(model, validation_iterator, criterion)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. F1: {str(valid_f1)} ')
    # report.write(f'Epoch {epoch} - Train Loss: {train_loss:.3f} | Val. F1: {str(valid_f1)}\n')

    df = pd.read_csv(TEST_PATH)
    df[f"{args['name']} ITER:{epoch}"] = [i + 1 for i in preds]
    # df.to_csv(TEST_PATH, index=False)

    # Showing statistics

# report.write('\n\n\n')
# report.close()



