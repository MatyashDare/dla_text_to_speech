import math
import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dmodel, dk):
        super().__init__()
        self.Qlayer = nn.Linear(dmodel, dk)
        self.Klayer = nn.Linear(dmodel, dk)
        self.Vlayer = nn.Linear(dmodel, dk)
        self.dk = dk

    def forward(self, x, attention_mask):
        Q = self.Qlayer(x)
        K = self.Klayer(x)
        V = self.Vlayer(x)
        QK = F.softmax(torch.matmul(Q, K.permute(0, 2, 1)) / self.dk ** 0.5 + attention_mask[:, :, None], dim=-1)
        return torch.matmul(QK, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel, dk, n_heads):
        super().__init__()
        self.head = nn.ModuleList()
        for i in range(n_heads):
            self.head.append(Attention(dmodel, dk))
        self.tail = nn.Linear(n_heads * dk, dmodel)

    def forward(self, x, attention_mask):
        tmp = []
        for module in self.head:
            tmp.append(module(x, attention_mask))
        out = torch.cat(tmp, dim=-1)
        out = self.tail(out)
        return out

    
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.model = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
       
    def forward(self, x):
        return self.model(x.permute(0, 2, 1)).permute(0, 2, 1)


class Layer(nn.Module):
    def __init__(self, dmodel, kernel, n_heads, dk):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(dmodel, dk, n_heads)
        self.middle_layer = nn.Sequential(Conv(dmodel, dmodel * 4, kernel[0]),
                                          nn.ReLU(), 
                                          Conv(dmodel * 4, dmodel, kernel[1])
                                         )
        self.norm = nn.LayerNorm(dmodel)
        self.layer_norm = nn.LayerNorm(dmodel)

    def forward(self, x, attention_mask):
        attention_output = self.multi_head_attention(x, attention_mask)
        y = self.middle_layer(x)
        x = self.norm (x + y)
        return x, attention_mask

    
def create_attention_mask(max_len, length):
    mask = (torch.arange(max_len)[None, :] > length[:, None]).float()
    mask[mask == 1] = -torch.inf
    return mask


def create_duplication(mel_len, cumsum, duration):
    a = (torch.arange(mel_len)[None, :] < (cumsum[:, None])).float()
    b = (torch.arange(mel_len)[None, :] >= (cumsum - duration)[:, None]).float()
    return a * b


def create_melspec(tokens, durations, mel_lengths):
    melspecs = []
    for i in range(tokens.shape[0]):
        tok = tokens[i]
        dur = durations[i]
        mel_len = mel_lengths[i]
        cumsum = dur.cumsum(0)
        melspec = torch.mm(create_duplication(mel_len, cumsum, dur).T, tok)
        melspecs.append(melspec)
    melspecs = nn.utils.rnn.pad_sequence(melspecs).permute(1, 0, 2)
    return melspecs


class DurationModel(nn.Module):
    def __init__(self, dmodel, kernel_size):
        super().__init__()
        self.duration_model = nn.Sequential(
                                Conv(dmodel, dmodel, kernel_size=kernel_size),
                                nn.LayerNorm(dmodel),
                                nn.ReLU(),
                                Conv(dmodel, dmodel, kernel_size=kernel_size),
                                nn.LayerNorm(dmodel),
                                nn.ReLU()
                            )
        self.linear = nn.Linear(dmodel, 1)

    def forward(self, x):
        y = self.duration_model(x)
        return self.linear(y).squeeze(2)


class PosEnc(nn.Module):
    def __init__(self, dmodel, max_len=5000):
        super(PosEnc, self).__init__()
        pe = torch.zeros(max_len, dmodel)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x   

    
class FSModel(nn.Module):
    def __init__(self, num_layers, dmodel, kernel, n_heads, dk):
        super().__init__()
        self.embedding = nn.Embedding(40, dmodel)
        self.duration_model = DurationModel(dmodel, 3)
        self.encoder = nn.ModuleList([Layer(dmodel, kernel, n_heads, dk) for i in range(num_layers)])
        self.decoder = nn.ModuleList([Layer(dmodel, kernel, n_heads, dk) for i in range(num_layers)])
        self.linear = nn.Linear(dmodel, 80)
        self.pos_enc = PosEnc(dmodel)
    
    def forward(self, tokens, token_lengths, durations):
        tokens = self.embedding(tokens)
        tokens = self.pos_enc(tokens)
        mask = create_attention_mask(tokens.shape[1], token_lengths)
        for i in range(len(self.encoder)):
            tokens, mask = self.encoder[i](tokens, mask)
        predicted_durations = self.duration_model(tokens)
        if durations is None:
            durations = (torch.exp(predicted_durations) - 1).round().int()
            durations[durations <= 1] = 1

        mel_lengths = durations.sum(1)
        melspecs = create_melspec(tokens, durations, mel_lengths)
        mask = create_attention_mask(melspecs.shape[1], mel_lengths)
        for l in self.encoder:
             melspecs, mask = l(melspecs, mask)
        return self.linear(melspecs), durations
