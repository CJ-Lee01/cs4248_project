import numpy as np
import pandas as pd
import torch
from torch import nn

if __name__ == '__main__':
    pass

import torch.nn as nn


class ClassificationLSTM(nn.Module):
    """
    The RNN model that will be used to perform classification.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(ClassificationLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


train_raw_data = pd.read_csv(f'./data/xtrain.txt', names=('label', 'text'), delimiter='\t')
test_raw_data = pd.read_csv(f'./data/balancedtest.csv', names=('label', 'text'))



from string import punctuation

train_texts = [''.join([c for c in text.lower() if c not in punctuation]) for text in train_raw_data['text']]
test_texts = [''.join([c for c in text.lower() if c not in punctuation]) for text in test_raw_data['text']]

# split by new lines and spaces
all_text_train = ' '.join(train_texts)
all_text_test = ' '.join(test_texts)

# create a list of words
words_train = all_text_train.split()
words_test = all_text_test.split()



from collections import Counter

# Build a dictionary that maps words to integers
counts = Counter(words_train)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

# use the dict to tokenize each article in text split
# store the tokenized texts in text_ints
text_ints_train = []
text_ints_test = []
for text in train_texts:
    text_ints_train.append([vocab_to_int[word] for word in text.split()])
for text in test_texts:
    text_ints_test.append([vocab_to_int[word] for word in text.split()])

# get the labels (0 and 1) from the data set
encoded_labels_train = [label for label in train_raw_data['label']]
encoded_labels_test = [label for label in test_raw_data['label']]

"""# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))

# print tokens in first article
print('Tokenized text: \n', text_ints[:1])"""

# outlier article stats
text_lens = Counter([len(x) for x in text_ints_train])
print("Zero-length text: {}".format(text_lens[0]))
print("Maximum text length: {}".format(max(text_lens)))


print('Number of texts before removing outliers: ', len(text_ints_train))

## remove any articles/labels with zero length from the text_ints list.

# get indices of any articles with length 0
non_zero_idx = [ii for ii, text in enumerate(text_ints_train) if len(text) != 0]

# remove 0-length articles and their labels
text_ints_train = [text_ints_train[ii] for ii in non_zero_idx]
encoded_labels_train = np.array([encoded_labels_train[ii] for ii in non_zero_idx])

print('Number of texts after removing outliers: ', len(text_ints_train))


def pad_features(text_ints, seq_length):
    ''' Return features of text_ints, where each article is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(text_ints), seq_length), dtype=int)

    # for each article, grab that article and
    for i, row in enumerate(text_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


seq_length = 200

features = pad_features(text_ints_train, seq_length=seq_length)

# test statements - do not change -
assert len(features)==len(text_ints_train), "Your features should have as many rows as articles."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches
print(features[:30, :10])



# split data into training, validation, and test data (features and labels, x and y)
split_idx = int(0.8 * len(train_raw_data)) # about 80%
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels_train[:split_idx], encoded_labels_train[split_idx:]

test_idx = (len(features)-16000)//2
val_x, test_x = remaining_x[:test_idx-2], remaining_x[test_idx-2:-4]
val_y, test_y = remaining_y[:test_idx-2], remaining_y[test_idx-2:-4]

# print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 10

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=5)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, num_workers=5)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=5)

# obtain one batch of training data
dataiter = iter(train_loader)
# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
output_size = 4
embedding_dim = 200
hidden_dim = 256
n_layers = 3

net = ClassificationLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
# move model to GPU, if available
if train_on_gpu:
    net.cuda()

print(net)

# loss and optimization functions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True)

# training params
epochs = 20
clip = 5  # gradient clipping
min_loss = np.inf

# train for some number of epochs
for e in range(epochs):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    num_correct = 0

    # initialize hidden state
    h = net.init_hidden(batch_size)

    net.train()
    # batch loop
    for inputs, labels in train_loader:

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * inputs.size(0)

    # Get validation loss
    val_h = net.init_hidden(batch_size)
    net.eval()
    for inputs, labels in valid_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        val_h = tuple([each.data for each in val_h])

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        output, val_h = net(inputs, val_h)
        loss = criterion(output.squeeze(), labels.float())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer
        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
        # update average validation loss
        valid_loss += loss.item() * inputs.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    scheduler.step(valid_loss)

    print("Epoch: {}/{}...".format(e + 1, epochs),
          "Loss: {:.6f}...".format(train_loss),
          "Val Loss: {:.6f}".format(valid_loss),
          "Accuracy: {:.6f}".format(num_correct / len(valid_loader.dataset)))
    if min_loss >= valid_loss:
        torch.save(net.state_dict(), 'checkpointx.pth')
        min_loss = valid_loss
        print("Loss decreased. Saving model...")

# Get test data loss and accuracy
test_losses = []  # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
loader = test_loader
# iterate over test data
for inputs, labels in loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if train_on_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()

    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(loader.dataset)
print("Test accuracy: {:.4f}%".format(test_acc * 100))


