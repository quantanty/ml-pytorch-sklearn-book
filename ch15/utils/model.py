import torch
from torch import nn

class RNN_v1(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, rnn_hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)

        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell
    
class RNN_v2(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, rnn_hidden_size=512, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(2*rnn_hidden_size)
        self.fc = nn.Linear(2*rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.layer_norm(out)
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(2, batch_size, self.rnn_hidden_size)
        return hidden, cell
    
class RNN_v3(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, rnn_hidden_size=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(2*rnn_hidden_size)
        self.fc = nn.Linear(2*rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.layer_norm(out)
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(2, batch_size, self.rnn_hidden_size)
        return hidden, cell