import torch
import torch.nn as nn
import torch.nn.functional as F

class SWEM_avg(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, add_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        # self.pooling = nn.AvgPool1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(add_dropout)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        embedded = embedded.permute(1,2,0)
        pooled = F.avg_pool1d(embedded ,embedded.shape[2])
        pooled.squeeze_()
        out = self.fc(pooled)
        return self.dropout(out)

class SWEM_max(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, add_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        # self.pooling = nn.AvgPool1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(add_dropout)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        embedded = embedded.permute(1,2,0)
        pooled = F.max_pool1d(embedded, embedded.shape[2])
        pooled.squeeze_()
        out = self.fc(pooled)
        return self.dropout(out)

class SWEM_hier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, add_dropout, n=5):
        super().__init__()
        self.kernel_size = n
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        # self.pooling = nn.AvgPool1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(add_dropout)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        embedded = embedded.permute(1,2,0)
        pooled = F.avg_pool1d(embedded, kernel_size=self.kernel_size)
        pooled = F.max_pool1d(pooled, pooled.shape[2])
        pooled.squeeze_()
        out = self.fc(pooled)
        return self.dropout(out)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.GRU(embedding_dim,
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        x, _ = self.rnn(embedded)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        x = x.permute((1, 2, 0))
        x = F.max_pool1d(x, x.shape[2]).squeeze()
        x = self.dropout(x)
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_channel, pad_idx, dp):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.conv = nn.Conv2d(1, out_channel, (3, embedding_dim))
        self.dropout = nn.Dropout(dp)
        self.fc = nn.Linear(out_channel, 1)

    def forward(self, x, x_length):
        x = self.embedding(x)
        x = x.permute((1, 0, 2))
        x = x.unsqueeze(1)
        x = F.relu(self.conv(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        x = self.dropout(x)
        return self.fc(x)