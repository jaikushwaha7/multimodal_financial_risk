import torch.nn as nn
import torch
import numpy as np

class CrossModalAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        Q = self.query(x1)
        K = self.key(x2)
        V = self.value(x2)
        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1)))
        return torch.matmul(attention_weights, V) + x1
