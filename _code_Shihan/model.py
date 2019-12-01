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