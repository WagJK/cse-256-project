import torch
import torch.nn as nn
import torch.nn.functional as F

class SWEM_avg(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        # self.pooling = nn.AvgPool1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        embedded = embedded.permute(1,2,0)
        pooled = F.avg_pool1d(embedded ,embedded.shape[2])
        pooled.squeeze_()
        return self.fc(pooled)

class SWEM_max(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        # self.pooling = nn.AvgPool1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        embedded = embedded.permute(1,2,0)
        pooled = F.max_pool1d(embedded, embedded.shape[2])
        pooled.squeeze_()
        return self.fc(pooled)

class SWEM_hier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, n=5):
        super().__init__()
        self.kernel_size = n
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        # self.pooling = nn.AvgPool1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        embedded = embedded.permute(1,2,0)
        pooled = F.avg_pool1d(embedded, kernel_size=self.kernel_size)
        pooled = F.max_pool1d(pooled, pooled.shape[2])
        pooled.squeeze_()
        return self.fc(pooled)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)