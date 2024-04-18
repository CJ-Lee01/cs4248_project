import torch
from torch import nn


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