import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RNNS = ['LSTM', 'GRU']


class ModelConfig:
    def __init__(self, name, marker, bi, md):
        self.name = name
        self.marker = marker
        self.bi = bi
        self.md = md

    def get_name(self):
        return self.name

    def get_marker(self):
        return self.marker

    def get_bi(self):
        return self.bi

    def get_md(self):
        return self.md


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.,
                 bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type)  # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(input_size, hidden_size, num_layers,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1)  # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0, 1).transpose(1, 2)  # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys)  # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize

        values = values.transpose(0, 1)  # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1)  # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


class Classifier(nn.Module):
    def __init__(self, encoder, attention, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, num_classes)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input, lengths):
        input = nn.utils.rnn.pack_padded_sequence(input, lengths=lengths, batch_first=True)
        outputs, hidden = self.encoder(input)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs.transpose(0, 1)
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]  # take the cell state

        if self.encoder.bidirectional:  # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)

        energy, linear_combination = self.attention(hidden, outputs, outputs)
        logits = self.decoder(hidden)
        return logits, energy
